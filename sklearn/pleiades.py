# Module containing personalized data analysis tools.

# Nabe deals with data cleaning and exploration.


class Nabe:

    def __init__(self):
        self.null_dict = None

    def get_nulls(self, df):
        self.null_dict = {}
        for k, v in df.isnull().sum().iteritems():
            if v > 0:
                self.null_dict[k] = v
        return self.null_dict

    # Drops columns with 75% or more null values.
    def drop_null_cols(self, df, null_size=0.75, inplace=False):
        if self.null_dict is None:
            raise Exception(r'use get_nulls(df) first.')
        if inplace is False:
            df = df.copy()
        df_size = df.shape[0]
        for k, v in self.null_dict.items():
            if null_size <= 1:
                if v/df_size >= null_size:
                    del df[k]
            else:
                if v >= null_size:
                    del df[k]
        return df

# CZ deals with word processing.
# She pastes 1 yen stickers on things she likes.


class CZ:

    def __init__(self):
        # Contractions dict.
        self.contractions = {
            "n't": " not",
            "'s": " is",
            "'m": " am",
            "'ll": " will",
            "'ve": " have",
            "'re": " are"
        }

    # To tokenize is to split the sentence into words.
    def re_tokenize(self, sentence):
        from nltk.tokenize import RegexpTokenizer
        retoken = RegexpTokenizer(r'\w+')
        sentence = sentence.lower()
        words = retoken.tokenize(sentence)
        return words

    # Lemmatizing eliminates things like the s from plurals like apples.
    def lemmatize_sentence(self, sentence):
        from nltk.stem import WordNetLemmatizer
        wnlem = WordNetLemmatizer()
        words = self.re_tokenize(sentence)
        words = [wnlem.lemmatize(word) for word in words]
        # Returns sentence instead of individual words.
        return ' '.join(words)

    # Stemming is a more drastic approach than lemmatizing. It truncates words
    # to get to the root word.
    def stem_sentence(self, sentence):
        from nltk.stem.porter import PorterStemmer
        p_stem = PorterStemmer()
        words = self.re_tokenize(sentence)
        words = [p_stem.stem(word) for word in words]
        # Returns sentence instead of individual words.
        return ' '.join(words)

    def text_list_cleaner(self, text_list, *args, replace=' ', inplace=False):
        '''
        Cleans text in lists.
        '''
        import re
        if inplace is False:
            text_list = text_list.copy()
        for i in range(len(text_list)):
            for arg in args:
                # Maps text with a function.
                if callable(arg):
                    text_list[i] = arg(text_list[i])
                # Maps text defined in dict keys with their corresponding
                # values.
                elif isinstance(arg, dict):
                    for k, v in arg.items():
                        text_list[i] = re.sub(k, v, text_list[i])
                # Removes all words passed as a list.
                elif not isinstance(arg, str):
                    for a in arg:
                        pattern = f' {a} '
                        text_list[i] = re.sub(pattern, replace, text_list[i])
                # For any other special cases.
                else:
                    text_list[i] = re.sub(arg, replace, text_list[i])
        return text_list

    def word_cloud(self, text, figsize=(12.5, 7.5), max_font_size=None, max_words=200, background_color='black', mask=None, recolor=False, **kwargs):
        '''
        Plots a wordcloud.

        Use full_text = ' '.join(list_of_text) to get a single string.
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        from wordcloud import WordCloud, ImageColorGenerator

        fig, ax = plt.subplots(figsize=figsize)

        if mask is None:
            cloud = WordCloud(background_color=background_color, max_words=max_words, **kwargs)
            cloud.generate(text)
            ax.imshow(cloud, interpolation='bilinear')
        else:
            m = np.array(Image.open(mask))
            cloud = WordCloud(background_color=background_color,
                              max_words=max_words, mask=m, **kwargs)
            cloud.generate(text)
            if recolor:
                image_colors = ImageColorGenerator(mask)
                ax.imshow(cloud.recolor(color_func=image_colors), interpolation='bilinear')
            else:
                ax.imshow(cloud, interpolation='bilinear')

        ax.axis('off')
        plt.show()
        plt.close()

# Solution handles feature selection and scaling.


class Solution:

    def __init__(self):
        pass

    def vif_feature_select(self, df, max_score=5.0, inplace=False, drop_list=False, _drops=None):
        '''
        Takes a DataFrame and returns it after recursively eliminating columns
        with the highest VIF scores until the remainder have a VIF scores of
        less than max_score.

        drop_list=True gets a list of features that would be dropped instead.
        '''
        import numpy as np

        # Avoids overwriting the original DataFrame by default.
        if inplace is False:
            df = df.copy()

        # Creates an empty list for the first iteration.
        if _drops is None:
            _drops = []

        features = df.columns
        # VIF is the diagonal of the correlation matrix.
        vifs = np.linalg.inv(df.corr().values).diagonal()
        max_vif_index = np.argmax(vifs)

        # Eliminate feature with the highest VIF score and rerun the function.
        if vifs[max_vif_index] >= max_score:
            _drops.append(features[max_vif_index])
            del df[features[max_vif_index]]
            return self.vif_feature_select(df, max_score, inplace, drop_list, _drops)
        else:
            # Returns a list of features that would be dropped instead of a
            # DataFrame
            if drop_list:
                return _drops
            else:
                return df

# Sebastian handles modeling.


class Sebastian:

    def __init__(self):
        self.feature_dict = None

    def get_params(self, dict):
        '''
        Formats the .best_params_ attribute of sklearn's models into a format
        that can be easily copy pasted onto the functions themselves.
        '''
        from re import match

        params = {}
        pattern = r'^([a-zA-Z0-9_]+)__([a-zA-Z0-9_]+)'
        for k, v in dict.items():

            if isinstance(v, str):
                v = "'" + v + "'"

            try:
                m = match(pattern, k)
                key = m.group(1)
                kwarg = f'{m.group(2)}={v}'
            except AttributeError:
                key = 'model args'
                kwarg = f'{k}={v}'

            if key in params:
                params[key].append(kwarg)
            else:
                params[key] = [kwarg]

        s = ''
        for k, v in params.items():
            joined_list = ', '.join(map(str, v))
            s += f'{k}: {joined_list} '

        return s.strip(' ')

    def get_features(self, X, feature_importances_, sort=True):
        '''
        Takes the train DataFrame and the .feature_importances_ attribute
        of sklearn's model and returns a sorted dictionary of
        feature_names: feature_importance for easy interpretation.
        '''
        feature_dict = {}
        for i, v in enumerate(feature_importances_):
            if v != 0:
                feature_dict[X.columns[i]] = v
        if sort:
            sorted_features = sorted(feature_dict, key=feature_dict.__getitem__, reverse=True)
            sorted_values = sorted(feature_dict.values(), reverse=True)
            sorted_feature_dict = {k: v for k, v in zip(sorted_features, sorted_values)}
            self.feature_dict = sorted_feature_dict
            return sorted_feature_dict
        else:
            self.feature_dict = feature_dict
            return feature_dict

        def plot_features(dict):
            import matplotlib.pyplot as plt
            features = dict.keys()
            importances = dict.values()
            plt.barh(range(len(features)), importances, align='center')
            plt.yticks(range(len(features)), features)
            plt.show()
            plt.close()

# Yuri handles data plots.


class Yuri:

    def __init__(self, y_test=None, y_prob=None):
        self.classes = None

    def auc_score(self, y_test, y_prob):
        '''
        A wrapper on the sklearn roc_auc_score that makes it work even if the
        target is multi-categorical or has not been label encoded.

        The auc_score normally ranges between 0.5 and 1. Less than 0.5 makes
        the model worse than the baseline.

        The function assumes y_prob is ordered in ascending order for the
        target.
        '''
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_auc_score

        classes = list(set(y_test))
        classes.sort()
        self.classes = classes
        n_classes = len(self.classes)
        is_multi_categorical = n_classes > 2

        # Avoids label_binarize if unnecessary.
        if is_multi_categorical:
            lb_test = label_binarize(y_test, classes=self.classes)
            auc_scores = []
            for i in range(n_classes):
                auc_scores.append(roc_auc_score(lb_test[:, i], y_prob[:, i]))
            # Returns the mean roc auc score.
            return auc_scores / n_classes
        else:
            try:
                y_prob = y_prob[:, 1]
                return roc_auc_score(y_test, y_prob)
            except IndexError:
                print('y_prob needs to have at least 2 columns.')
            except TypeError:
                lb_test = label_binarize(y_test, classes=self.classes)
                return roc_auc_score(lb_test, y_prob)

    def dt_auc_scores(self, X_train, X_test, y_train, y_test, param_grid, tree='dt', **kwargs):
        '''
        Returns the AUROC scores for the 3 most important parameters of a
        decision tree. It is used in conjunction with plot_auc to help
        visualize decision tree parameters.
        '''
        if tree == 'dt':
            from sklearn.tree import DecisionTreeClassifier
            dt_type = DecisionTreeClassifier
        elif tree == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            dt_type = RandomForestClassifier
        elif tree == 'et':
            from sklearn.ensemble import ExtraTreesClassifier
            dt_type = ExtraTreesClassifier
        else:
            raise Exception('unrecognized tree type.')

        train_auc_scores = []
        test_auc_scores = []
        for key, value in param_grid.items():
            for v in value:
                if key == 'max_depth' or key == 'md':
                    dt = dt_type(max_depth=v, **kwargs)
                elif key == 'min_samples_split' or key == 'mss':
                    dt = dt_type(min_samples_split=v, **kwargs)
                elif key == 'min_samples_leaf' or key == 'msl':
                    dt = dt_type(min_samples_leaf=v, **kwargs)
                else:
                    raise Exception('unrecognized param.')
                dt.fit(X_train, y_train)

                y_prob_train = dt.predict_proba(X_train)
                train_auc_scores.append(self.auc_score(y_train, y_prob_train))

                y_prob = dt.predict_proba(X_test)
                test_auc_scores.append(self.auc_score(y_test, y_prob))

        return [train_auc_scores, test_auc_scores]

    def plot_auc(self, x, auc_scores, lw=2, title=None, xlabel=None, labels=None, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(**kwargs)
        for i, scores in enumerate(auc_scores):
            if labels is None:
                labels = range(len(auc_scores))
            ax.plot(x, scores, label=labels[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('AUC score')
        ax.set_title(title)
        ax.legend(loc='best')
        plt.show()
        plt.close()

    def plot_roc(self, y_test, y_prob, average='macro', mm=False, reverse_classes=False, lw=2, title=None, labels=None, **kwargs):
        '''
        Plots Receiver Operating Characteristic (ROC) curves for predict_proba
        method for sklearn models.

        This function is built to make plotting of ROC curves for a model with
        multi-categorical targets painless. It takes the one vs all approach
        when plotting the ROC curve for each target class.

        params:
            average 'macro' accepts 3 possible arguments besides None. 'macro',
                    'micro' or 'both'. It determines whether and what kind of
                    mean ROC curve to plot for multi-categorical targets.
            mm      If set to True, makes the function capable of plotting
                    ROC curves of multiple binary target models in the same
                    figure. It will cause the function to treat y_prob as a
                    list of y_probs instead of the y_prob of a single model.
                    mm stands for multi model.
            labels  accepts a dictionary of column values mapped onto class
                    names. If the column values are simply integers, it is
                    possible to just pass a list.

        The function assumes y_prob is ordered in ascending order for the
        target.
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        from itertools import cycle
        from scipy import interp

        # Gets all unique categories.
        classes = list(set(y_test))
        classes.sort()
        self.classes = classes
        is_multi_categorical = len(self.classes) > 2
        lb_test = label_binarize(y_test, classes=self.classes)

        # Initialize graph.
        fig, ax = plt.subplots(**kwargs)

        if is_multi_categorical:

            # Compute ROC curve and ROC area for each class.
            fpr = {}
            tpr = {}
            roc_auc = {}
            for i, k in enumerate(self.classes):
                fpr[k], tpr[k], _ = roc_curve(lb_test[:, i], y_prob[:, i])
                roc_auc[k] = auc(fpr[k], tpr[k])

            if average == 'micro' or average == 'both':
                # Compute micro-average ROC curve and ROC area.
                fpr['micro'], tpr['micro'], _ = roc_curve(
                    lb_test.ravel(), y_prob.ravel())
                roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

                ax.plot(fpr['micro'], tpr['micro'], ':r',
                        label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})', lw=lw)

            if average == 'macro' or average == 'both':
                # Compute macro-average ROC curve and ROC area.

                # First aggregate all false positive rates.
                all_fpr = np.unique(np.concatenate(
                    [fpr[k] for k in self.classes]))

                # Then interpolate all ROC curves at these points.
                mean_tpr = np.zeros_like(all_fpr)
                for k in self.classes:
                    mean_tpr += interp(all_fpr, fpr[k], tpr[k])

                # Finally average it and compute AUC.
                mean_tpr /= len(self.classes)

                fpr['macro'] = all_fpr
                tpr['macro'] = mean_tpr
                roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

                ax.plot(fpr['macro'], tpr['macro'], ':b',
                        label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})', lw=lw)

            # Plot ROC curve for each category.
            colors = cycle(['teal', 'darkorange', 'cornflowerblue'])
            if labels is None:
                labels = self.classes
            for k, color in zip(self.classes, colors):
                ax.plot(fpr[k], tpr[k], color=color,
                        label=f'ROC curve of {labels[k]} (area = {roc_auc[k]:0.2f})', lw=lw)

        else:

            def plot_single_roc(lb_test, y_prob, labels, i='target'):
                y_prob = y_prob[:, 1]
                fpr, tpr, _ = roc_curve(lb_test, y_prob)
                roc_auc = auc(fpr, tpr)
                if labels is None:
                    labels = f'class {i}'
                else:
                    labels = labels[i]
                ax.plot(
                    fpr, tpr, label=f'ROC curve of {labels} (area = {roc_auc:0.2f})', lw=lw)

            # Allows plotting of multiple binary target ROC curves in the same
            # figure.
            if mm:
                for i in range(len(y_prob)):
                    plot_single_roc(lb_test, y_prob[i], labels, i)
            else:
                plot_single_roc(lb_test, y_prob, labels)

        # Plot the curve of the baseline model (mean).
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc='best')
        plt.show()
        plt.close()
