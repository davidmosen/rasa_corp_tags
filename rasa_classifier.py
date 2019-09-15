import numpy as np
import pandas as pd

import os
import shutil
import wget
import tempfile

import fasttext

from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer
from rasa_nlu.model import Interpreter
from rasa_nlu.test import run_evaluation

from sklearn.base import BaseEstimator, ClassifierMixin
import sklearn.metrics


class RasaLangClassifier:
    """
    The core Rasa classifier, specific to one language. One instance of the
    language corresponds to one Rasa model, with the language and configuration
    indicated in the input configuration string.
    """

    def __init__(self, config_str, base_dir=None, verbose=0):
        self.config_str = config_str
        self.verbose = verbose
        self.base_dir = base_dir
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.project_name = 'current'
        self.model_name = 'nlu'
        self.last_model_dir = os.path.join(
            self.models_dir,
            self.project_name,
            self.model_name
        )
        self.nlu_file = os.path.join(self.data_dir, 'nlu.md')
        self.config_file = os.path.join(self.base_dir, 'config.yml')

        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        for d in [self.data_dir, self.last_model_dir]:
            os.makedirs(d)
        if self.verbose > 0:
            print("Successfully created base directory structure {}".format(
                base_dir
            ))

        # write configuration to config file
        with open(self.config_file, "w") as text_file:
            print(config_str, file=text_file)

    def fit(self, X, y):
        # format data into Rasa NLU markdown file format
        df_data = pd.DataFrame({'text': X})
        df_data['intent'] = y.apply(
            lambda i: '+'.join(i.index[i > 0]), axis=1
        )
        series_intents = df_data.groupby('intent')['text'].apply(
            lambda texts: '## intent:' + texts.name + '\n' + '\n'.join(
                ['- ' + t for t in texts]
            )
        )
        intents = '\n\n'.join(series_intents)
        with open(self.nlu_file, "w") as text_file:
            print(intents, file=text_file)
        self.tags = df_data['intent'].unique()

        training_data = load_data(self.nlu_file)
        trainer = Trainer(config.load(self.config_file))
        trainer.train(training_data)
        model_directory = trainer.persist(
            self.models_dir,
            project_name=self.project_name,
            fixed_model_name=self.model_name
        )

        self.interpreter = Interpreter.load(self.last_model_dir)

    def predict_conf(self, X):
        def predict_conf_single(self, question):
            # get confidence from interpreter
            try:
                interp_output = self.interpreter.parse(question)
                intent_ranking = interp_output['intent_ranking']
            except AttributeError as error:
                raise AttributeError(
                    'The model needs to be trained first.'
                ) from error
            out = pd.Series(0, index=self.tags)
            df_intents = pd.DataFrame.from_dict(intent_ranking)
            out[df_intents['name']] = df_intents['confidence']
            # return a pd.Series()
            return(out)

        X_ = [X] if np.isscalar(X) else X
        out = pd.DataFrame([predict_conf_single(self, Xi) for Xi in X_])
        return(out)

    def predict(self, X):
        def predict_single(self, question):
            s_conf = self.predict_conf(question)
            indiv_tags = list(set([
                item
                for sublist in [t.split('+') for t in self.tags]
                for item in sublist
            ]))
            s = pd.Series(0.0, index=indiv_tags)
            s[s_conf.iloc[0].idxmax().split('+')] = 1
            return(s)

        X_ = [X] if np.isscalar(X) else X
        return(pd.DataFrame([predict_single(self, Xi) for Xi in X_]))

    def get_params(self, deep=True):
        return({'config_str': self.config_str})


class RasaClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn-compatible class that can handle questions in any of the
    176 languages supported by the fastText language model. Effectively, it
    acts as a wrapper to as many RasaLangClassifier objects as needed, calling
    them accordingly.
    """

    def __init__(
        self,
        config_str='pipeline: "supervised_embeddings"',
        base_dir=None,
        verbose=0
    ):
        self.config_str = config_str
        if base_dir is None:
            self.base_dir = tempfile.mkdtemp()
        else:
            self.base_dir = base_dir
        self.verbose = verbose

    def fit(self, X, y=None):
        self.langs = [
            lang.replace('lang:', '')
            for lang in y.filter(regex='^lang:').columns
        ]
        self.lang_dirs = {
            lang: os.path.join(self.base_dir, lang)
            for lang in self.langs
        }
        self.lang_classifiers = {}
        for lang in self.langs:
            self.lang_classifiers[lang] = RasaLangClassifier(
                'language: {lang}\n'.format(lang=lang) + self.config_str,
                self.lang_dirs[lang],
                verbose=self.verbose
            )
        for lang, cls in self.lang_classifiers.items():
            cls.fit(
                X[y['lang:' + lang] > 0],
                y[y['lang:' + lang] > 0].filter(regex='^[^lang:]')
            )
        return(self)

    def _internal_predict(self, X, func_name, **kwargs):
        def check_langs(det_lang):
            unknown_langs = list(set(det_lang).difference(set(self.langs)))
            if len(unknown_langs) > 0:
                raise IndexError(
                    'Unsupported langs detected:{unk}. Avail:{langs}.'.format(
                        unk=unknown_langs, langs=self.langs
                    )
                )

        X_ = [X] if np.isscalar(X) else X

        # NOTE this'd be part of the prediction package in final implementation
        try:
            lang_model = fasttext.load_model('lid.176.ftz')
        except:
            f = 'https://dl.fbaipublicfiles.com/' + \
                'fasttext/supervised-models/lid.176.ftz'
            wget.download(f)
            lang_model = fasttext.load_model('lid.176.ftz')
        det_lang = np.array([
            lang_model.predict(q)[0][0].replace('__label__', '')
            for q in X_
        ])
        check_langs(det_lang)
        out = pd.DataFrame()
        ids = []
        for lang in np.sort(list(set(det_lang))):
            func = getattr(self.lang_classifiers[lang], func_name)
            out_l = func(np.array(X_)[det_lang == lang], **kwargs)
            ids = np.concatenate([ids, np.where(det_lang == lang)[0]])
            out = out.append(out_l, ignore_index=True)
        lang_dummies = pd.get_dummies(pd.Series([
            'lang:' + lang
            for lang in np.sort(det_lang)
        ]))
        out = pd.concat([lang_dummies, out], axis=1).iloc[np.argsort(ids)]
        return(out)

    def predict_conf(self, X):
        return(self._internal_predict(X, 'predict_conf', **{}))

    def predict(self, X):
        return(self._internal_predict(X, 'predict', **{}))

    def score(self, X, y):
        y_hat = self._internal_predict(X, 'predict', **{})
        s = sklearn.metrics.f1_score(
            np.array(y, dtype='float'),
            np.array(y_hat[y.columns], dtype='float'),
            average='samples'
        )
        return(s)

    def get_params(self, deep=True):
        return({'config_str': self.config_str})
