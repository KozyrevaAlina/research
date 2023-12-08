import pickle
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from federated_learning.dataset_old import prepare_dataset ###
from federated_learning.client import generate_client_fn
from federated_learning.server import get_on_fit_config, get_evaluate_fn
from federated_learning.model import BiLSTM

# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
# Декоратор для Гидры. Это говорит гидре о необходимости по умолчанию загружать конфигурацию в conf/base.yaml.
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    ## 1. Разобрать конфигурацию и получить выходной каталог эксперимента
    print(OmegaConf.to_yaml(cfg))
    # Hydra automatically creates a directory for your experiments
    # by default it would be in <this directory>/outputs/<date>/<time>
    # you can retrieve the path to it as shown below. We'll use this path to
    # save the results of the simulation (see the last part of this main())

    # Hydra автоматически создает каталог для ваших экспериментов, по умолчанию 
    # он будет находиться в <этот каталог>/outputs/<дата>/<время>, вы можете получить 
    # путь к нему, как показано ниже. Мы будем использовать этот путь для сохранения 
    # результатов моделирования (см. последнюю часть этого main()).
    save_path = HydraConfig.get().runtime.output_dir

    ## 2. Prepare your dataset
    # When simulating FL workloads we have a lot of freedom on how the FL clients behave,
    # what data they have, how much data, etc. This is not possible in real FL settings.
    # In simulation you'd often encounter two types of dataset:
    #       * naturally partitioned, that come pre-partitioned by user id (e.g. FEMNIST,
    #         Shakespeare, SpeechCommands) and as a result these dataset have a fixed number
    #         of clients and a fixed amount/distribution of data for each client.
    #       * and others that are not partitioned in any way but are very popular in ML
    #         (e.g. MNIST, CIFAR-10/100). We can _synthetically_ partition these datasets
    #         into an arbitrary number of partitions and assign one to a different client.
    #         Synthetically partitioned dataset allow for simulating different data distribution
    #         scenarios to tests your ideas. The down side is that these might not reflect well
    #         the type of distributions encounter in the Wild.
    #
    # In this tutorial we are going to partition the MNIST dataset into 100 clients (the default
    # in our config -- but you can change this!) following a independent and identically distributed (IID)
    # sampling mechanism. This is arguably the simples way of partitioning data but it's a good fit
    # for this introductory tutorial.

    ## 2. Подготовьте набор данных
    # При моделировании рабочих нагрузок FL у нас есть большая свобода в отношении того, 
    # как ведут себя клиенты FL, какие данные они имеют, сколько данных и т. д. Это невозможно в реальных настройках FL.
    # При моделировании часто встречаются два типа наборов данных:
    # * естественно разделенные, которые предварительно разделены по идентификатору пользователя 
    # (например, FEMNIST, Shakespeare, SpeechCommands), и в результате этот набор данных имеет 
    # фиксированное количество клиентов и фиксированный объем/распределение данных для каждого клиента.
    # * и другие, которые никак не секционированы, но очень популярны в ML (например, MNIST, CIFAR-10/100). 
    # Мы можем _синтетически_ разделить эти наборы данных на произвольное количество секций и назначить одну из них другому клиенту.
    # Синтетически секционированный набор данных позволяет моделировать различные сценарии распределения 
    # данных для проверки ваших идей. Обратной стороной является то, что они могут не совсем хорошо отражать тип дистрибутивов, встречающихся в дикой природе.
    #
    # В этом уроке мы собираемся разделить набор данных MNIST на 100 клиентов (по умолчанию в нашей конфигурации — но вы можете это изменить!), 
    # используя независимый и одинаково распределенный (IID) механизм выборки. Возможно, это самый простой способ разделения данных, 
    # но он хорошо подходит для этого вводного руководства.
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )

    ## 3. Define your clients
    # Unlike in standard FL (e.g. see the quickstart-pytorch or quickstart-tensorflow examples in the Flower repo),
    # in simulation we don't want to manually launch clients. We delegate that to the VirtualClientEngine.
    # What we need to provide to start_simulation() with is a function that can be called at any point in time to
    # create a client. This is what the line below exactly returns.

    ## 3. Определите своих клиентов
    # В отличие от стандартного FL (например, см. примеры faststart-pytorch или faststart-tensorflow в репозитории Flower), 
    # в симуляции мы не хотим запускать клиенты вручную. Мы делегируем это VirtualClientEngine.
    # Что нам нужно предоставить в start_simulation(), так это функцию, которую можно вызвать в 
    # любой момент времени для создания клиента. Именно это возвращает строка ниже.
    client_fn = generate_client_fn(Net, trainloaders, validationloaders, cfg.num_classes) ####################

    ## 4. Define your strategy
    # A flower strategy orchestrates your FL pipeline. Although it is present in all stages of the FL process
    # each strategy often differs from others depending on how the model _aggregation_ is performed. This happens
    # in the strategy's `aggregate_fit()` method. In this tutorial we choose FedAvg, which simply takes the average
    # of the models received from the clients that participated in a FL round doing fit().
    # You can implement a custom strategy to have full control on all aspects including: how the clients are sampled,
    # how updated models from the clients are aggregated, how the model is evaluated on the server, etc
    # To control how many clients are sampled, strategies often use a combination of two parameters `fraction_{}` and `min_{}_clients`
    # where `{}` can be either `fit` or `evaluate`, depending on the FL stage. The final number of clients sampled is given by the formula
    # ``` # an equivalent bit of code is used by the strategies' num_fit_clients() and num_evaluate_clients() built-in methods.
    #         num_clients = int(num_available_clients * self.fraction_fit)
    #         clients_to_do_fit = max(num_clients, self.min_fit_clients)
    # ```

    ## 4. Определите свою стратегию
    # Flower стратегия управляет вашим конвейером FL. Хотя она присутствует на всех этапах процесса FL,
    # каждая стратегия часто отличается от других в зависимости от того, как выполняется агрегация модели. 
    # Это происходит в методе Aggregate_fit() стратегии. В этом уроке мы выбираем FedAvg, который просто 
    # берет среднее значение моделей, полученных от клиентов, участвовавших в раунде FL, выполняющем fit().

    # Вы можете реализовать собственную стратегию, чтобы иметь полный контроль над всеми аспектами, включая: 
    # как осуществляется выборка клиентов, как агрегируются обновленные модели клиентов, как модель оценивается на сервере и т. д.
    # Чтобы контролировать количество выбранных клиентов, стратегии часто используют комбинацию двух параметров 
    # `fraction_{}` и `min_{}_clients`, где `{}` может быть либо `fit`, либо `evaluate`, в зависимости от стадии FL . 
    # Окончательное количество отобранных клиентов определяется по формуле
    # ``` # эквивалентный фрагмент кода используется встроенными методами стратегий num_fit_clients() и num_evaluate_clients().
    # num_clients = int(num_available_clients * self.fraction_fit)
    # client_to_do_fit = max(num_clients, self.min_fit_clients)
    # ```
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        min_fit_clients=cfg.num_clients_per_round_fit,  # number of clients to sample for fit()
        fraction_evaluate=0.0,  # similar to fraction_fit, we don't need to use this argument.
        min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
        min_available_clients=cfg.num_clients,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        # функция, которую необходимо выполнить для получения конфигурации для отправки клиентам во время fit()
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )  # a function to run on the server side to evaluate the global model.
       # функция, запускаемая на стороне сервера для оценки глобальной модели.

    ## 5. Start Simulation
    # With the dataset partitioned, the client function and the strategy ready, we can now launch the simulation!

    ## 5. Запустить симуляцию
    # Когда набор данных разделен, клиентская функция и стратегия готовы, мы можем запустить моделирование!
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client # функция, которая порождает конкретного клиента
        num_clients=cfg.num_clients,  # total number of clients # общее количество клиентов
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL # минимальная конфигурация для цикла сервера, сообщающая количество раундов в FL
        strategy=strategy,  # our strategy of choice # наша стратегия выбора
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.0,
        },  # (optional) controls the degree of parallelism of your simulation.
        # Lower resources per client allow for more clients to run concurrently
        # (but need to be set taking into account the compute/memory footprint of your workload)
        # `num_cpus` is an absolute number (integer) indicating the number of threads a client should be allocated
        # `num_gpus` is a ratio indicating the portion of gpu memory that a client needs.

        # (необязательно) управляет степенью параллелизма вашей симуляции.
         # Меньшее количество ресурсов на клиент позволяет одновременно запускать больше клиентов (но это необходимо настраивать с учетом объема вычислений и памяти вашей рабочей нагрузки)
         # `num_cpus` — абсолютное число (целое), указывающее количество потоков, которые должен выделить клиент
         # `num_gpus` — это коэффициент, указывающий долю памяти графического процессора, необходимую клиенту.
    )

    # ^ Following the above comment about `client_resources`. if you set `num_gpus` to 0.5 and you have one GPU in your system,
    # then your simulation would run 2 clients concurrently. If in your round you have more than 2 clients, then clients will wait
    # until resources are available from them. This scheduling is done under-the-hood for you so you don't have to worry about it.
    # What is really important is that you set your `num_gpus` value correctly for the task your clients do. For example, if you are training
    # a large model, then you'll likely see `nvidia-smi` reporting a large memory usage of you clients. In those settings, you might need to
    # leave `num_gpus` as a high value (0.5 or even 1.0). For smaller models, like the one in this tutorial, your GPU would likely be capable
    # of running at least 2 or more (depending on your GPU model.)
    # Please note that GPU memory is only one dimension to consider when optimising your simulation. Other aspects such as compute footprint
    # and I/O to the filesystem or data preprocessing might affect your simulation  (and tweaking `num_gpus` would not translate into speedups)
    # Finally, please note that these gpu limits are not enforced, meaning that a client can still go beyond the limit initially assigned, if
    # this happens, your might get some out-of-memory (OOM) errors.

    ## 6. Save your results
    # (This is one way of saving results, others are of course valid :) )
    # Now that the simulation is completed, we could save the results into the directory
    # that Hydra created automatically at the beginning of the experiment.

    # ^ После комментария выше о `client_resources`. если вы установите для `num_gpus` значение 0,5 и в вашей системе есть один графический процессор, 
    # тогда ваша симуляция будет запускать 2 клиента одновременно. Если в вашем раунде у вас более 2 клиентов, то клиенты будут ждать, пока от них не 
    # станут доступны ресурсы. Это планирование выполняется за вас «под капотом», поэтому вам не нужно об этом беспокоиться.
    
    # Что действительно важно, так это правильно установить значение `num_gpus` для задачи, которую выполняют ваши клиенты. 
    # Например, если вы обучаете большую модель, вы, скорее всего, увидите, что nvidia-smi сообщает о большом использовании памяти вашими клиентами. 
    # В этих настройках вам может потребоваться оставить для `num_gpus` высокое значение (0,5 или даже 1,0). Для моделей меньшего размера, таких как модель, 
    # описанная в этом руководстве, ваш графический процессор, скорее всего, сможет работать как минимум с двумя или более (в зависимости от модели 
    # вашего графического процессора).
    # Обратите внимание, что память графического процессора — это только одно измерение, которое следует учитывать при оптимизации моделирования. 
    # Другие аспекты, такие как вычислительный объем и ввод-вывод в файловую систему или предварительная обработка данных, могут повлиять на ваше моделирование 
    # (и настройка `num_gpus` не приведет к ускорению)
    # Наконец, обратите внимание, что эти ограничения графического процессора не применяются, а это означает, что клиент все равно может выйти за 
    # пределы изначально назначенного предела. Если это произойдет, вы можете получить некоторые ошибки нехватки памяти (OOM).

    ## 6. Сохраните результаты
    # (Это один из способов сохранения результатов, другие, конечно, допустимы :) )
    # Теперь, когда моделирование завершено, мы можем сохранить результаты в каталоге, который Hydra автоматически создала в начале эксперимента.
    results_path = Path(save_path) / "results.pkl"

    # add the history returned by the strategy into a standard Python dictionary
    # you can add more content if you wish (note that in the directory created by
    # Hydra, you'll already have the config used as well as the log)

    # добавить историю, возвращаемую стратегией, в стандартный словарь Python, при желании вы можете добавить 
    # больше контента (обратите внимание, что в каталог, созданный Гидра, у вас уже есть используемый конфиг и лог)
    results = {"history": history, "anythingelse": "here"}

    # # сохраняем результаты как pickle Python 
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
