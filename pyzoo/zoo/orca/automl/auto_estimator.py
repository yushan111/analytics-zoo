#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


class AutoEstimator:
    def __init__(self, model_builder, searcher):
        self.model_builder = model_builder
        self.searcher = searcher
        self._fitted = False

    @staticmethod
    def from_torch(*,
                   model_creator,
                   optimizer,
                   loss,
                   logs_dir,
                   resources_per_trial,
                   name,
                   ):
        """
        Create an AutoEstimator for torch.

        :param model_creator: PyTorch model creator function.
        :param optimizer: PyTorch optimizer creator function or pytorch optimizer name (string).
            Note that you should specify learning rate search space with key as "lr" or LR_NAME
            (from zoo.orca.automl.pytorch_utils import LR_NAME) if input optimizer name.
            Without learning rate search space specified, the default learning rate value of 1e-3
            will be used for all estimators.
        :param loss: PyTorch loss instance or PyTorch loss creator function
            or pytorch loss name (string).
        :param logs_dir: Local directory to save logs and results.
        :param resources_per_trial: Dict. resources for each trial. e.g. {"cpu": 2}.
        :param name: Name of the auto estimator.
        :return: an AutoEstimator object.
        """
        from zoo.orca.automl.pytorch_utils import validate_pytorch_loss, \
            validate_pytorch_optim
        from zoo.automl.model import ModelBuilder
        from zoo.automl.search import SearchEngineFactory
        loss = validate_pytorch_loss(loss)
        optimizer = validate_pytorch_optim(optimizer)
        model_builder = ModelBuilder.from_pytorch(model_creator=model_creator,
                                                  optimizer_creator=optimizer,
                                                  loss_creator=loss)
        searcher = SearchEngineFactory.create_engine(backend="ray",
                                                     logs_dir=logs_dir,
                                                     resources_per_trial=resources_per_trial,
                                                     name=name)
        return AutoEstimator(model_builder=model_builder, searcher=searcher)

    @staticmethod
    def from_keras(*,
                   model_creator,
                   logs_dir,
                   resources_per_trial,
                   name,
                   ):
        """
        Create an AutoEstimator for tensorflow keras.

        :param model_creator: Tensorflow keras model creator function.
        :param logs_dir: Local directory to save logs and results.
        :param resources_per_trial: Dict. resources for each trial. e.g. {"cpu": 2}.
        :param name: Name of the auto estimator.
        :return: an AutoEstimator object.
        """
        from zoo.automl.model import ModelBuilder
        from zoo.automl.search import SearchEngineFactory
        model_builder = ModelBuilder.from_tfkeras(model_creator=model_creator)
        searcher = SearchEngineFactory.create_engine(backend="ray",
                                                     logs_dir=logs_dir,
                                                     resources_per_trial=resources_per_trial,
                                                     name=name)
        return AutoEstimator(model_builder=model_builder, searcher=searcher)

    def fit(self,
            data,
            recipe=None,
            metric=None,
            search_alg=None,
            search_alg_params=None,
            scheduler=None,
            scheduler_params=None,
            ):
        if self._fitted:
            raise RuntimeError("auto_estimator.fit is not expected to be called multiple times.")
        self.searcher.compile(data=data,
                              model_create_func=self.model_builder,
                              recipe=recipe,
                              metric=metric,
                              search_alg=search_alg,
                              search_alg_params=search_alg_params,
                              scheduler=scheduler,
                              scheduler_params=scheduler_params)
        self.searcher.run()
        self._fitted = True

    def get_best_model(self):
        best_trial = self.searcher.get_best_trials(k=1)[0]
        best_model_path = best_trial.model_path
        best_model = self.model_builder.build_from_ckpt(best_model_path)
        return best_model
