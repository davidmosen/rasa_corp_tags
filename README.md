# Prediction of tags associated to questions about limited company type and shareholders

## Highlights

The multi-label multi-language classifer that:
* Can handle questions in any of 176 languages.
* Extends scikit-learn BaseEstimator: it is fully compatible with scikit-learn's API.
* Take in any tag name and combination of tags.
* Implement any Rasa pipeline.
* Can also report individual confidence of tag combinations (*predict_conf()* function).
* Is as simple to use as:
```
cls = RasaClassifier()
cls.fit(X_train, y_train)
cls.predict(X_test)
cls.predict_conf(X_test)
```

## Setting up the environment

If using Conda, the following commands need to be run to set up the environment:
```
conda create -n rasa_corp_tags python=3.6 -y --no-default-packages
# the following command might be necessary
# export PATH="/home/<user>/anaconda3/bin:$PATH"
source activate rasa_corp_tags
# in production, versions will need to be specified to avoid compatibility issues
pip install --user ipykernel fasttext wget numpy pandas scikit-learn rasa-nlu sklearn-crfsuite tensorflow
python -m ipykernel install --user --name=rasa_corp_tags
```

## Files

The Jupyter notebook *notebook.ipynb* contains the code to make use of the core functionality, which is inside *rasa_classifier.py*.

The CSV file *co_sh_questions.csv* contains a set of manually annotated questions, scraped from Google's "People Also Ask" section using Chrome extension [Scraper](https://chrome.google.com/webstore/detail/scraper/mbigbapnjcgaffohmbkdlecaccepngjd).

The additional notebook *lang_detection.ipynb* assesses how well different Python language-detection packages do on the questions in the CSV.

## Next steps

* Improvement of the configuration (e.g. setting up synonyms and/or regex).
* More extensive docstrings and documentation.
* Creation of a package.
* Setting up an API.
* Unit testing and continuous integration.
