# Module containing personalized data cleaning and exploration tools.

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

    def text_list_cleaner(self, text_list, *args, inplace=False):
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
                        pattern = f'\b{a}\b'
                        text_list[i] = re.sub(pattern, r'', text_list[i])
                # For any other special cases.
                else:
                    text_list[i] = re.sub(arg, r' ', text_list[i])
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

        # Sebastian deals with data cleaning.


class Sebastian:

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