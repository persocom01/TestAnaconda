# Module containing personalized data analysis tools.

# CZ deals with databases.
# She primarily functions to make the use of sqlalchemy easier.
# She pastes 1 yen stickers on things she likes.


class CZ:

    def __init__(self, engine=None, database=None):
        self.engine = engine
        self.database = database
        self.dtype_dic = {
            'int64': 'INT',
            'float64': 'DOUBLE',
            'bool': 'BOOLEAN',
            'datetime64': 'DATETIME'
        }
        self.tabspace = 4

    # This function is meant to be used on boto3.resource objects.
    def get_keys(self, bucket_name, prefix='/', suffix=None, delimiter='/'):
        import re
        prefix = prefix[1:] if prefix.startswith(delimiter) else prefix
        bucket = self.engine.Bucket(bucket_name)
        keys = [_.key for _ in bucket.objects.filter(Prefix=prefix)]
        for key in keys:
            if suffix:
                if not re.search(suffix, key):
                    keys.remove(key)
            if key[-1:] == delimiter:
                keys.remove(key)
        return keys

    # This function is meant to be used on boto3.resource objects.
    def download_files(self, bucket_name, keys, savein=''):
        import re
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            filename = re.search(r'/([^/]+)$', key, re.I)[1]
            file_path = savein + filename
            bucket = self.engine.Bucket(bucket_name)
            bucket.download_file(key, file_path)
        return f'{len(keys)} files downloaded.'

    class SQL:
        '''
        The SQL object allows a SQL statement to be extended with methods like
        where() before being executed with .ex()
        '''

        def __init__(self, command, engine=None, tabspace=4):
            self.command = command
            self.engine = engine
            self.tabspace = tabspace

        def ex(self, p=False):
            command = self.command
            if p or self.engine is None:
                return command
            import pandas as pd
            df = pd.read_sql_query(command, self.engine)
            return df

        def where(self, condition):
            command = self.command
            command = command[:-1] + f'WHERE {condition}\n;'
            self.command = command
            return self

    def mk_db(self, db, charset='utf8', collate='utf8_general_ci', printable=False):
        command = f'CREATE DATABASE {db} CHARACTER SET {charset} COLLATE {collate};'
        if printable or self.engine is None:
            return command
        from sqlalchemy.exc import ProgrammingError
        try:
            self.engine.connect().execute(command)
            return f'database {db} created.'
        except ProgrammingError as err:
            return err

    def show_db(self, printable=False):
        command = f'SHOW DATABASES;'
        if printable or self.engine is None:
            return command
        import pandas as pd
        df = pd.read_sql_query(command, self.engine)
        return df

    def del_db(self, db, printable=False):
        command = f'DROP DATABASE {db};'
        if printable or self.engine is None:
            return command
        from sqlalchemy.exc import InternalError
        try:
            self.engine.connect().execute(command)
            return f'database {db} deleted.'
        except InternalError as err:
            return err

    # Returns the name of the currently selected db.
    def current_db(self, printable=False):
        command = 'SELECT DATABASE();'
        if printable or self.engine is None:
            return command
        return self.engine.connect().execute(command).fetchone()[0]

    def use_db(self, db=None, printable=False):
        if db is None:
            db = self.database
        command = f'USE {db};'
        if printable or self.engine is None:
            return command
        self.engine.connect().execute(command)
        return f'database {db} selected.'

    def unuse_db(self, printable=False, _db='2arnbzheo2j0gygkteu9ltxtabmzldvb'):
        command = f'CREATE DATABASE {_db};'
        command += f'\nUSE {_db};'
        command += f'\nDROP DATABASE {_db};'
        if printable or self.engine is None:
            return command
        self.engine.connect().execute(command)
        return 'database deselected.'

    def select_from(self, table, cols=None):
        tab = ' ' * self.tabspace
        command = 'SELECT'
        if cols:
            if isinstance(cols, str):
                command += f' {cols}\n'
            else:
                command += f'\n{tab}'
                for col in cols:
                    command += f'{col}\n{tab},'
                command = command[:-(self.tabspace+1)]
        else:
            command += f' *\n'
        if self.database:
            table = self.database + '.' + table
        command += f'FROM {table}\n;'
        return self.SQL(command, engine=self.engine)

    def csv_clean_colnames(self, file, sep=''):
        '''
        Given the path to a csv file, this function cleans the column names by
        converting removing leading and trailing white spaces, converting all
        letters to lowercase, replacing all remaining whitespaces with
        underscores, removing brackets, forward slashes, and other special
        characters. The csv file is then replaced with a copy of itself with
        the cleaned column names.
        params:
            file        path of file wholse column names are to be cleaned.
            sep         The character(s) used to replace brackets and special
                        characters.
        '''
        import re
        import csv
        import os

        def remove_special_characters(text, sep=sep):
            pattern = r'[^a-zA-Z0-9!"#$%&\'()*+, -./:; <= >?@[\]^_`{|}~]'
            return re.sub(pattern, sep, text)

        # Opens the csv file and writes the cleaned version to a .tmp file.
        tempfile = file + '.tmp'
        with open(file, 'r') as infile, open(tempfile, 'w', newline='') as outfile:
            r = csv.reader(infile, delimiter=',', quotechar='"')
            colnames = next(r)
            colnames = [remove_special_characters(x.strip().lower().replace(' ', '_').replace(
                '(', sep).replace(')', sep).replace('/', sep)) for x in colnames]
            w = csv.writer(outfile)
            w.writerow(colnames)
            for row in r:
                w.writerow(row)

        # Delete original and replace it with the cleaned file.
        os.remove(file)
        os.rename(tempfile, file)

    def csv_table(self, file, table=None, pkey=None, nrows=100000, printable=False, **kwargs):
        '''
        Creates an empty table based on data from a file. Normally unnecessary
        as pandas .to_sql() creates the table automatically, but could be
        useful when that doesn't work.
        params:
            file        file the table datatypes will be based on.
            table       if None, table = filename.
            pkeys       the table names to pass on when defining the PRIMARY
                        KEYs of the table. If a list is passed, a composite
                        primary key will be defined.
            nrows       determines the number of rows read by pandas when
                        imputing the table datatypes. A large value results in
                        unnecessary data being read. A small value may result
                        in incorrect table datatype values.
            printable   returns the SQL command that would have been executed
                        as a printable string.
            **kwargs    Other arguments to be passed on to pandas read_csv.
        '''
        import pandas as pd
        from pathlib import Path
        from math import ceil
        from sqlalchemy.exc import InternalError
        # pandas is used to impute datatypes.
        df = pd.read_csv(file, nrows=nrows, **kwargs)
        tab = ' ' * self.tabspace
        # The file name will be used as the table name if not provided.
        if table is None:
            table = Path(file).stem
        if self.database:
            table = self.database + '.' + table

        def get_sql_dtypes(df):
            sql_dtype_dict = {}
            df_dtypes = [x for x in df.dtypes.apply(lambda x: x.name)]
            df = df.fillna('')
            # pandas dtypes are converted to sql dtypes to create the table.
            for i, col in enumerate(df.columns):
                if df_dtypes[i] in self.dtype_dic:
                    sql_dtype_dict[col] = self.dtype_dic[df_dtypes[i]]
                else:
                    # Determine VARCHAR length.
                    char_length = ceil(df[col].map(len).max() / 50) * 50
                    sql_dtype_dict[col] = f'VARCHAR({char_length})'
            return sql_dtype_dict

        command = f'CREATE TABLE {table}(\n{tab}'
        sql_dtype_dict = get_sql_dtypes(df)
        for col, sql_dtype in sql_dtype_dict.items():
            command = command + f'{col} {sql_dtype}\n{tab},'
        if pkey:
            if isinstance(pkey, str):
                pkey = [pkey]
            pkey = ', '.join(pkey)
            command += f'PRIMARY KEY({pkey})\n{tab},'
        command = command[:-(self.tabspace+1)] + ');'
        if printable or self.engine is None:
            return command
        try:
            self.engine.connect().execute(command)
            return f'table {table} created.'
        except InternalError as err:
            return err

    def csv_insert(self, file, table=None, pkey=None, postgre=False, chunksize=None, sizelim=1073741824, printable=False, **kwargs):
        '''
        Convenience function that uploads file data into a database.
        params:
            file        path of file to be uploaded.
            pkey        given the table's PRIMARY KEY, the function updates all
                        values in the table with those from the file except the
                        primary key. If a new table is created, pkey is
                        set as the table's primary key.
            postgre     set to True if working on a PostgreSQL database. Only
                        relevant if not using sqlalchemy.
            table       if None, table = filename.
            chunksize   determines the number of rows read from the csv file to
                        insert into the database at a time. This is
                        specifically meant to deal with memory issues. As such,
                        when chunksize != None and printable == True, the
                        commands will be written to chunk_insert.txt instead of
                        being returned for printing.
            sizelim     determines the file size, in bytes, before a default
                        chunksize of 10000 is imposed if chunksize is not
                        already specified.
            printable   returns the SQL command that would have been executed
                        as a printable string. It doesn't work well past a few
                        thousand rows or so.
            **kwargs    Other arguments to be passed on to pandas read_csv.
        '''
        from pathlib import Path
        from re import sub
        from sqlalchemy.exc import InternalError
        from sqlalchemy.exc import IntegrityError
        import pandas as pd
        if table is None:
            table = Path(file).stem
        # Automatically set chunksize if file exceeds sizelim.
        if Path(file).stat().st_size >= sizelim and chunksize is None:
            chunksize = 100000
        if self.database:
            self.csv_table(file, table=table, pkey=pkey)
            table = self.database + '.' + table

        def individual_insert(df, table=None):
            rows = [x for x in df.itertuples(index=False, name=None)]
            cols = ', '.join(df.columns)
            tab = ' ' * self.tabspace
            for r in rows:
                command = f'INSERT INTO {table}({cols}) VALUES '
                # Fix null values.
                pattern = r"([^\w'])nan([^\w'])"
                replacement = r'\1NULL\2'
                fixed_r = sub(pattern, replacement, f'{r}')
                command += f'{fixed_r}\n'
                if pkey:
                    if postgre:
                        command = command[:-(self.tabspace+1)] + \
                            f'ON CONFLICT ({pkey}) DO UPDATE SET\n{tab}'
                        for c in df.columns:
                            if c != pkey:
                                command += f'{c}=excluded.{c}\n{tab},'
                    else:
                        command += f'ON DUPLICATE KEY UPDATE\n{tab}'
                        for c in df.columns:
                            if c not in pkey:
                                command += f'{c}=VALUES({c})\n{tab},'
                    command = command[:-(self.tabspace+1)] + ';'
                try:
                    self.engine.connect().execute(command)
                except (InternalError, IntegrityError):
                    continue

        def alchemy_insert(df, pkey=None, table=None):
            try:
                df.to_sql(table, self.engine, index=False, if_exists='append')
            except (InternalError, IntegrityError):
                individual_insert(df, table=table)
            if pkey:
                try:
                    command = f'ALTER TABLE {table} ADD PRIMARY KEY({pkey});'
                    self.engine.connect().execute(command)
                except InternalError as err:
                    return err

        def mass_insert(df, table=None, pkey=None, postgre=False):
            rows = [x for x in df.itertuples(index=False, name=None)]
            cols = ', '.join(df.columns)
            tab = ' ' * self.tabspace
            command = f'INSERT INTO {table}({cols})\nVALUES\n{tab}'
            for r in rows:
                # Fix null values.
                pattern = r"([^\w'])nan([^\w'])"
                replacement = r'\1NULL\2'
                fixed_r = sub(pattern, replacement, f'{r}')
                command += f'{fixed_r}\n{tab},'
            if pkey:
                if postgre:
                    command = command[:-(self.tabspace+1)] + \
                        f'ON CONFLICT ({pkey}) DO UPDATE SET\n{tab}'
                    for c in df.columns:
                        if c != pkey:
                            command += f'{c}=excluded.{c}\n{tab},'
                else:
                    command = command[:-(self.tabspace+1)] + \
                        f'ON DUPLICATE KEY UPDATE\n{tab}'
                    for c in df.columns:
                        if c != pkey:
                            command += f'{c}=VALUES({c})\n{tab},'
            command = command[:-(self.tabspace+1)] + ';\n'
            return command

        if chunksize:
            reader = pd.read_csv(file, chunksize=chunksize, **kwargs)
            for chunk in reader:
                df = pd.DataFrame(chunk)
                if printable:
                    with open('chunk_insert.txt', 'a') as f:
                        f.write(mass_insert(df, pkey=pkey,
                                            postgre=postgre, table=table))
                else:
                    alchemy_insert(df, pkey=pkey, table=table)
            if printable:
                return 'sql commands written to chunk_insert.txt'
            else:
                return f'data loaded into table {table}.'

        else:
            df = pd.read_csv(file, **kwargs)
            if printable:
                return mass_insert(df, pkey=pkey, postgre=postgre, table=table)
            alchemy_insert(df, pkey=pkey, table=table)
            return f'data loaded into table {table}.'

    def csvs_into_database(self, file_paths, table=None, clean_colnames=False, pkeys=None, **kwargs):
        '''
        Convenience function that uploads a folder of files into a database.
        params:
            file_paths      a string passed to the glob module which determines
                            what files to to upload. Normally in the format
                            './folder/*.extension'
            table       the table to upload the files into. Use when all
                            files are to be uploaded into a SINGLE TABLE.
            clean_colnames  standardizes and gets rid of potentially
                            problematic characters in column names. Warning:
                            replaces the column names in the original file.
            pkeys           accepts a list of PRIMARY KEYs to be assigned to
                            each table to be created. Must be given in file
                            alphabetical order as the files will be read in
                            alphabetical order. The list can be incomplete, but
                            any table not assigned a primary key should have ''
                            as its corresponding list item. Note that
                            alphabetical on atom can be different from the way
                            python processes the files if _ is in the file
                            name.
            **kwargs        optional arguments passed to pandas read_csv
                            function. na_values can be specified,
                            keep_default_na=False, low_memory=False are useful
                            arguments.
        '''
        import glob
        files = glob.glob(file_paths)
        has_incomplete_pkeys = False
        if pkeys:
            if isinstance(pkeys, str):
                pkeys = [pkeys]
            if table:
                for i, file in enumerate(files):
                    if clean_colnames:
                        self.csv_clean_colnames(file)
                    try:
                        self.csv_insert(file, table=table,
                                        pkey=pkeys[i], **kwargs)
                    except TypeError or IndexError:
                        has_incomplete_pkeys = True
                        self.csv_insert(file, table=table, **kwargs)
            else:
                for i, file in enumerate(files):
                    if clean_colnames:
                        self.csv_clean_colnames(file)
                    try:
                        self.csv_insert(file, pkey=pkeys[i], **kwargs)
                    except TypeError or IndexError:
                        has_incomplete_pkeys = True
                        self.csv_insert(file, **kwargs)
        else:
            if table:
                for file in files:
                    if clean_colnames:
                        self.csv_clean_colnames(file)
                    self.csv_insert(file, table=table, **kwargs)
            else:
                for file in files:
                    if clean_colnames:
                        self.csv_clean_colnames(file)
                    self.csv_insert(file, **kwargs)

        return_statement = f'files written to database {self.current_db()}.'
        if has_incomplete_pkeys:
            return_statement = 'not all tables have primary keys.\n' + return_statement
        return return_statement

    def show_tables(self, all=False, printable=False):
        if self.database:
            if all:
                command = f"SELECT TABLE_NAME FROM {self.database}.INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE';"
            else:
                command = f'SHOW TABLES FROM {self.database};'
        else:
            if all:
                command = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE';"
            else:
                command = 'SHOW TABLES;'
        if printable or self.engine is None:
            return command
        import pandas as pd
        df = pd.read_sql_query(command, self.engine)
        return df

    def clone_table(self, target, new_table=None, cols=None, where=None, printable=False):
        from sqlalchemy.exc import InternalError
        if new_table is None:
            new_table = target + '_copy'
        command = f'CREATE TABLE {new_table} AS\n'
        if self.database:
            target = self.database + '.' + target
        if cols:
            if isinstance(cols, str):
                cols = [cols]
            cols = ', '.join(cols)
            command += f'SELECT {cols}\n'
        else:
            command += f'SELECT *\n'
        command += f'FROM {target}\n'
        if where:
            command += f'WHERE {where}\n;'
        else:
            command = command + ';'
        if printable or self.engine is None:
            return command
        try:
            self.engine.connect().execute(command)
        except InternalError as err:
            return err
        return f'table {target} cloned into table {new_table}.'

    def del_tables(self, tables, printable=False):
        from sqlalchemy.exc import InternalError
        tab = ' ' * self.tabspace
        command = 'DROP TABLES'
        if isinstance(tables, str):
            tables = [tables]
        if self.database:
            for i, table in enumerate(tables):
                tables[i] = self.database + '.' + table
        if len(tables) == 1:
            command += f' {tables};'
            return_string = tables
        else:
            command += f'\n{tab}'
            for table in tables:
                command += f'{table}\n{tab},'
            command = command[:-(self.tabspace+1)] + ';'
        if printable or self.engine is None:
            return command
        try:
            self.engine.connect().execute(command)
        except InternalError as err:
            return err
        return_string = ', '.join(tables)
        return f'table(s) {return_string} deleted.'

    def show_columns(self, table, all=False, printable=False):
        if self.database:
            table = self.database + '.' + table
        if all:
            command = f'SHOW ALL COLUMNS FROM {table};'
        else:
            command = f'SHOW COLUMNS FROM {table};'
        if printable or self.engine is None:
            return command
        import pandas as pd
        df = pd.read_sql_query(command, self.engine)
        return df

    def insert_columns(self, to_table, from_table, cols=None, where=None, printable=False):
        from sqlalchemy.exc import InternalError
        if self.database:
            to_table = self.database + '.' + to_table
        command = f'INSERT INTO {to_table}\n'
        if cols is None:
            cols = '*'
        elif isinstance(cols, str):
            cols = [cols]
            cols = ', '.join(cols)
        else:
            cols = ', '.join(cols)
        command += f'SELECT {cols}\nFROM {from_table}\n'
        if where:
            command += f'WHERE {where}\n;'
        else:
            command += ';'
        if printable or self.engine is None:
            return command
        try:
            self.engine.connect().execute(command)
        except InternalError as err:
            return err
        return_string = ', '.join(cols)
        return f'column(s) {return_string} inserted into {to_table} from {from_table}.'

    def del_columns(self, table, cols, if_exists=True, printable=False):
        from sqlalchemy.exc import InternalError
        if self.database:
            table = self.database + '.' + table
        command = f'ALTER TABLE {table}\n'
        if isinstance(cols, str):
            cols = [cols]
        if if_exists:
            drop_statement = 'DROP COLUMN IF EXISTS'
        else:
            drop_statement = 'DROP COLUMN'
        for col in cols:
            command += f'{drop_statement} {col}\n'
        command += ';'
        if printable or self.engine is None:
            return command
        try:
            self.engine.connect().execute(command)
        except InternalError as err:
            return err
        return_string = ', '.join(cols)
        return f'column(s) {return_string} deleted.'


# Nabe deals with data cleaning and exploration.


class Nabe:

    def __init__(self):
        self.null_dict = None
        self.steps = '''
        1. df.head()
        2. df.info()
        3. df.isnull().sum() or 1 - df.count() / df.shape[0]
        4. clean
        5. visualize correlations
        '''

    def get_null_indexes(self, df, cols=None):
        '''
        Takes a DataFrame and returns a dictionary of columns and the row
        indexes of the null values in them.
        '''
        # Prevents errors from passing a string instead of a list.
        if isinstance(cols, str):
            cols = [cols]

        null_indexes = []
        null_dict = {}
        if cols is None:
            cols = df.columns
        for col in cols:
            null_indexes = df[df[col].isnull()].index.tolist()
            null_dict[col] = null_indexes
        return null_dict

    # Drops columns with 75% or more null values.
    def drop_null_cols(self, df, null_size=0.75, inplace=False):
        if inplace is False:
            df = df.copy()
        null_table = 1 - df.count() / df.shape[0]
        non_null_cols = [i for i, v in enumerate(null_table) if v < null_size]
        df = df.iloc[:, non_null_cols]
        return df

    # Returns the row index of a column value.
    def get_index(self, df, col_name, value):
        if len(df.loc[df[col_name] == value]) == 1:
            return df.loc[df[col_name] == value].index[0]
        else:
            return df.loc[df[col_name] == value].index

# Lupusregina deals with word processing.


class Lupu:

    def __init__(self):
        # Contractions dict.
        self.contractions = {
            "n't": " not",
            "n’t": " not",
            "'s": " is",
            "’s": " is",
            "'m": " am",
            "’m": " am",
            "'ll": " will",
            "’ll": " will",
            "'ve": " have",
            "’ve": " have",
            "'re": " are",
            "’re": " are"
        }
        self.re_ref = {
            'email': r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',
            'link': r'(https?://[^ ]+)',
            'gender_pronoun': [r'[hH]e/[hH]im', '[tT]hey/[tT]hem', '[tT]ey/[tT]em', '[eE]y/[eE]m', '[eE]/[eE]m', '[tT]hon/[tT]hon', '[fF]ae/[fF]aer', '[vV]ae/[vV]aer', '[aA]e/[aA]er', '[nN]e/[nN]ym', '[nN]e/[nN]em', '[xX]e/[xX]em', '[xX]e/[xX]im', '[xX]ie/[xX]em', '[zZ]e/[zZ]ir', '[zZ]ie/[zZ]ir', '[zZ]he/[zZ]hir', '[zZ]e/[hH]ir', '[sS]ie/[sS]ier', '[zZ]ed/[zZ]ed', '[zZ]ed/[zZ]ed', '[cC]e/[cC]ir', '[cC]o/[cC]os', '[vV]e/[vV]is', '[jJ]ee/[jJ]em', '[lL]ee/[lL]im', '[kK]ye/[kK]yr', '[pP]er/[pP]er', '[hH]u/[hH]um', '[bB]un/[bB]un', '[iI]t/[iI]t']
        }
        self.sep = ' '

    # Lowercase.
    def to_lower(self, sentence):
        return sentence.lower()

    # To tokenize is to split the sentence into words.
    def re_tokenize(self, sentence, sep=r'\w+'):
        from nltk.tokenize import RegexpTokenizer
        retoken = RegexpTokenizer(sep)
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

    def remove_punctuation(self, sentence, sep=None):
        import re
        if sep is None:
            sep = self.sep
        sentence = re.sub(
            r'[!"#$%&\'()*+, -./:; <= >?@[\]^_`{|}~’“”]', sep, sentence)
        return sentence

    def split_camel_case(self, sentence):
        import re
        splitted = re.sub('([A-Z][a-z]+)', r' \1',
                          re.sub('([0-9A-Z]+)', r' \1', sentence)).split()
        return ' '.join(splitted)

    def text_list_cleaner(self, text_list, *args, sep=None, inplace=False):
        '''
        Function made to make chain transformations on text lists easy.

        Maps words when passed functions or dictionaries as *arguments.
        Removes words when passed lists or strings.

        Aside from lists, all *arguments should be made in regex format, as the
        function does not account for spaces or word boundaries by default.
        '''
        import re
        if inplace is False:
            text_list = text_list.copy()
        if sep is None:
            sep = self.sep

        # Prevents KeyError from passing a pandas series with index not
        # beginning in 0.
        try:
            iter(text_list.index)
            r = text_list.index
        except TypeError:
            r = range(len(text_list))

        for i in r:
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
                        pattern = r'\b{}\b'.format(a)
                        text_list[i] = re.sub(pattern, sep, text_list[i])
                # For any other special cases.
                else:
                    text_list[i] = re.sub(arg, sep, text_list[i])
        return text_list

    def word_cloud(self, text, figsize=(12.5, 7.5), max_font_size=None, max_words=200, background_color='black', mask=None, recolor=False, export_path=None, **kwargs):
        '''
        Plots a wordcloud.

        Use full_text = ' '.join(list_of_text) to get a single string.
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        from wordcloud import WordCloud, ImageColorGenerator

        fig, ax = plt.subplots(figsize=figsize)
        if mask:
            m = np.array(Image.open(mask))
            cloud = WordCloud(background_color=background_color,
                              max_words=max_words, mask=m, **kwargs)
            cloud.generate(text)
            if recolor:
                image_colors = ImageColorGenerator(mask)
                ax.imshow(cloud.recolor(color_func=image_colors),
                          interpolation='bilinear')
            else:
                ax.imshow(cloud, interpolation='bilinear')
        else:
            cloud = WordCloud(background_color=background_color,
                              max_words=max_words, **kwargs)
            cloud.generate(text)
            ax.imshow(cloud, interpolation='bilinear')
        if export_path:
            cloud.to_file(export_path)
        ax.axis('off')
        plt.show()
        plt.close()


# Nabe deals with data cleaning and exploration.


class Nabe:

    def __init__(self):
        self.null_dict = None
        self.steps = '''
        1. df.head()
        2. df.info()
        3. df.isnull().sum() or 1 - df.count() / df.shape[0]
        4. clean
        5. visualize correlations
        '''

    def get_null_indexes(self, df, cols=None):
        '''
        Takes a DataFrame and returns a dictionary of columns and the row
        indexes of the null values in them.
        '''
        # Prevents errors from passing a string instead of a list.
        if isinstance(cols, str):
            cols = [cols]

        null_indexes = []
        null_dict = {}
        if cols is None:
            cols = df.columns
        for col in cols:
            null_indexes = df[df[col].isnull()].index.tolist()
            null_dict[col] = null_indexes
        return null_dict

    # Drops columns with 75% or more null values.
    def drop_null_cols(self, df, null_size=0.75, inplace=False):
        if inplace is False:
            df = df.copy()
        null_table = 1 - df.count() / df.shape[0]
        non_null_cols = [i for i, v in enumerate(null_table) if v < null_size]
        df = df.iloc[:, non_null_cols]
        return df

    # Returns the row index of a column value.
    def get_index(self, df, col_name, value):
        if len(df.loc[df[col_name] == value]) == 1:
            return df.loc[df[col_name] == value].index[0]
        else:
            return df.loc[df[col_name] == value].index

# Lupusregina deals with word processing.


class Lupu:

    def __init__(self):
        # Contractions dict.
        self.contractions = {
            "n't": " not",
            "n’t": " not",
            "'s": " is",
            "’s": " is",
            "'m": " am",
            "’m": " am",
            "'ll": " will",
            "’ll": " will",
            "'ve": " have",
            "’ve": " have",
            "'re": " are",
            "’re": " are"
        }
        self.re_ref = {
            'email': r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',
            'link': r'(https?://[^ ]+)',
            'gender_pronoun': [r'[hH]e/[hH]im', '[tT]hey/[tT]hem', '[tT]ey/[tT]em', '[eE]y/[eE]m', '[eE]/[eE]m', '[tT]hon/[tT]hon', '[fF]ae/[fF]aer', '[vV]ae/[vV]aer', '[aA]e/[aA]er', '[nN]e/[nN]ym', '[nN]e/[nN]em', '[xX]e/[xX]em', '[xX]e/[xX]im', '[xX]ie/[xX]em', '[zZ]e/[zZ]ir', '[zZ]ie/[zZ]ir', '[zZ]he/[zZ]hir', '[zZ]e/[hH]ir', '[sS]ie/[sS]ier', '[zZ]ed/[zZ]ed', '[zZ]ed/[zZ]ed', '[cC]e/[cC]ir', '[cC]o/[cC]os', '[vV]e/[vV]is', '[jJ]ee/[jJ]em', '[lL]ee/[lL]im', '[kK]ye/[kK]yr', '[pP]er/[pP]er', '[hH]u/[hH]um', '[bB]un/[bB]un', '[iI]t/[iI]t']
        }
        self.sep = ' '

    # Lowercase.
    def to_lower(self, sentence):
        return sentence.lower()

    # To tokenize is to split the sentence into words.
    def re_tokenize(self, sentence, sep=r'\w+'):
        from nltk.tokenize import RegexpTokenizer
        retoken = RegexpTokenizer(sep)
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

    def remove_punctuation(self, sentence, sep=None):
        import re
        if sep is None:
            sep = self.sep
        sentence = re.sub(
            r'[!"#$%&\'()*+, -./:; <= >?@[\]^_`{|}~’“”]', sep, sentence)
        return sentence

    def split_camel_case(self, sentence):
        import re
        splitted = re.sub('([A-Z][a-z]+)', r' \1',
                          re.sub('([0-9A-Z]+)', r' \1', sentence)).split()
        return ' '.join(splitted)

    def text_list_cleaner(self, text_list, *args, sep=None, inplace=False):
        '''
        Function made to make chain transformations on text lists easy.

        Maps words when passed functions or dictionaries as *arguments.
        Removes words when passed lists or strings.

        Aside from lists, all *arguments should be made in regex format, as the
        function does not account for spaces or word boundaries by default.
        '''
        import re
        if inplace is False:
            text_list = text_list.copy()
        if sep is None:
            sep = self.sep

        # Prevents KeyError from passing a pandas series with index not
        # beginning in 0.
        try:
            iter(text_list.index)
            r = text_list.index
        except TypeError:
            r = range(len(text_list))

        for i in r:
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
                        pattern = r'\b{}\b'.format(a)
                        text_list[i] = re.sub(pattern, sep, text_list[i])
                # For any other special cases.
                else:
                    text_list[i] = re.sub(arg, sep, text_list[i])
        return text_list

    def word_cloud(self, text, figsize=(12.5, 7.5), max_font_size=None, max_words=200, background_color='black', mask=None, recolor=False, export_path=None, **kwargs):
        '''
        Plots a wordcloud.

        Use full_text = ' '.join(list_of_text) to get a single string.
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        from wordcloud import WordCloud, ImageColorGenerator

        fig, ax = plt.subplots(figsize=figsize)
        if mask:
            m = np.array(Image.open(mask))
            cloud = WordCloud(background_color=background_color,
                              max_words=max_words, mask=m, **kwargs)
            cloud.generate(text)
            if recolor:
                image_colors = ImageColorGenerator(mask)
                ax.imshow(cloud.recolor(color_func=image_colors),
                          interpolation='bilinear')
            else:
                ax.imshow(cloud, interpolation='bilinear')
        else:
            cloud = WordCloud(background_color=background_color,
                              max_words=max_words, **kwargs)
            cloud.generate(text)
            ax.imshow(cloud, interpolation='bilinear')
        if export_path:
            cloud.to_file(export_path)
        ax.axis('off')
        plt.show()
        plt.close()

# Solution handles feature selection and scaling.


class Solution:

    def __init__(self):
        pass

    def cramers_corr(self, df):
        '''
        Takes a DataFrame of categorical variables and returns a DataFrame of
        the correlation matrix based on the Cramers V statistic. Uses
        correction from Bergsma and Wicher, Journal of the Korean Statistical
        Society 42 (2013): 323-328

        Does not require variables to be label encoded before use.
        '''
        import numpy as np
        import pandas as pd
        import scipy.stats as stats
        from itertools import combinations

        def cramers_v(x, y):
            con_table = pd.crosstab(x, y)
            chi2 = stats.chi2_contingency(con_table)[0]
            n = con_table.sum().sum()
            phi2 = chi2/n
            r, k = con_table.shape
            phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
            rcorr = r-((r-1)**2)/(n-1)
            kcorr = k-((k-1)**2)/(n-1)
            return np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))

        cols = df.columns
        n_cols = len(cols)
        corr_matrix = np.zeros((n_cols, n_cols))
        for col1, col2 in combinations(cols, 2):
            i1, i2 = cols.get_loc(col1), cols.get_loc(col2)
            corr_matrix[i1, i2] = cramers_v(df[col1], df[col2])
            corr_matrix[i2, i1] = corr_matrix[i1, i2]
        np.fill_diagonal(corr_matrix, 1.0)
        df_corr_matrix = pd.DataFrame(corr_matrix, index=cols, columns=cols)
        return df_corr_matrix

    def vif_feature_select(self, df, max_score=5.0, inplace=False, printable=False, _drops=None):
        '''
        Takes a DataFrame and returns it after recursively eliminating columns
        with the highest VIF scores until the remainder have VIF scores less
        than max_score.

        params:
            printable   when set to True, the function returns a list of
                        features that would be dropped instead.
        '''
        import numpy as np
        # Avoids overwriting the original DataFrame by default.
        if inplace is False:
            df = df.copy()
        # Creates an empty list for the first iteration.
        if _drops is None:
            _drops = []
            # Check if features contain string values.
            dtypes = [dt for dt in df.dtypes]
            if 'object' in dtypes:
                print('Feature(s) contain string values. Result may be unreliable.')
            # Check if any feature contains all 0s.
            if (df == 0).all().any():
                raise Exception(
                    'Feature(s) contain all 0s. Drop them before use.')
        features = df.columns
        # VIF is the diagonal of the correlation matrix.
        vifs = np.linalg.inv(df.corr().values).diagonal()
        max_vif_index = np.argmax(vifs)
        # Eliminate feature with the highest VIF score and rerun the function.
        if vifs[max_vif_index] >= max_score:
            _drops.append(features[max_vif_index])
            del df[features[max_vif_index]]
            return self.vif_feature_select(df, max_score, inplace, printable, _drops)
        else:
            # Returns a list of features that would be dropped instead of a
            # DataFrame
            if printable:
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
            # Puts quotes on string argument values.
            if isinstance(v, str):
                v = "'" + v + "'"
            # Checks if params are from a pipeline.
            try:
                m = match(pattern, k)
                key = m.group(1)
                kwarg = f'{m.group(2)}={v}'
            # For non pipeline params.
            except AttributeError:
                key = 'model args'
                kwarg = f'{k}={v}'
            # Populates dictionary with step: params.
            if key in params:
                params[key].append(kwarg)
            else:
                params[key] = [kwarg]
        # Turns dictionary into string for easy copy paste.
        s = ''
        for k, v in params.items():
            joined_list = ', '.join(map(str, v))
            s += f'{k}: {joined_list} '
        return s.strip(' ')

    def get_features(self, X_train, feature_importances_, order=None):
        '''
        Takes the train DataFrame and the .feature_importances_ attribute of
        sklearn's model and returns a sorted dictionary of feature_names:
        feature_importance for easy interpretation.
        '''
        # Creates feature dict of features with non zero importances.
        feature_dict = {}
        for i, v in enumerate(feature_importances_):
            if v != 0:
                feature_dict[X_train.columns[i]] = v
        # Sorts dict from most important feature to least.
        if order == 'dsc':
            sorted_features = sorted(
                feature_dict, key=feature_dict.__getitem__, reverse=True)
            sorted_values = sorted(feature_dict.values(), reverse=True)
            sorted_feature_dict = {k: v for k, v in zip(
                sorted_features, sorted_values)}
        elif order == 'asc':
            sorted_features = sorted(
                feature_dict, key=feature_dict.__getitem__, reverse=False)
            sorted_values = sorted(feature_dict.values(), reverse=False)
            sorted_feature_dict = {k: v for k, v in zip(
                sorted_features, sorted_values)}
        elif order == 'abs':
            feature_dict = {k: abs(v) for k, v in feature_dict.items()}
            sorted_features = sorted(
                feature_dict, key=feature_dict.__getitem__, reverse=True)
            sorted_values = sorted(feature_dict.values(), reverse=True)
            sorted_feature_dict = {k: v for k, v in zip(
                sorted_features, sorted_values)}
        else:
            sorted_feature_dict = feature_dict
        self.feature_dict = sorted_feature_dict
        return sorted_feature_dict

    def plot_importances(self, X_train=None, feature_importances_=None, max_features=10, order='dsc', fontsize=10, title=None, **kwargs):
        '''
        Takes the train DataFrame and the .feature_importances_ attribute of
        sklearn's model and plots a horizontal bar graph of the 10 most
        important features and their importances.

        Can be called without any arguments if get_features() was called
        beforehand.

        params:
            max_features    determines the number of features plotted. The
                            default is 10.
            order           'des' plots features with the highest importances.
                            'asc' plots features with the lowest importances.
                            This can be useful if importances have -ve values.
                            'abs' takes the absolute value of feature
                            importances before plotting those with the highest
                            values.
        '''
        import matplotlib.pyplot as plt
        # Allows the function to be called after get_features with no
        # arguments.
        if X_train is None or feature_importances_ is None:
            if self.get_features is None:
                raise TypeError(
                    'missing "X_train" or "feature_importances_" arguments.')
            else:
                feature_dict = self.feature_dict
        else:
            feature_dict = self.get_features(
                X_train, feature_importances_, sort=False)
            self.feature_dict = feature_dict
        # Arranges the graph from most important at the top to least at the
        # bottom.
        if order == 'dsc':
            sorted_features = sorted(
                feature_dict, key=feature_dict.__getitem__, reverse=True)
            sorted_values = sorted(feature_dict.values(), reverse=True)
            sorted_feature_dict = {k: v for k, v in zip(
                sorted_features, sorted_values)}
        elif order == 'asc':
            sorted_features = sorted(
                feature_dict, key=feature_dict.__getitem__, reverse=False)
            sorted_values = sorted(feature_dict.values(), reverse=False)
            sorted_feature_dict = {k: v for k, v in zip(
                sorted_features, sorted_values)}
        elif order == 'abs':
            feature_dict = {k: abs(v) for k, v in feature_dict.items()}
            sorted_features = sorted(
                feature_dict, key=feature_dict.__getitem__, reverse=True)
            sorted_values = sorted(feature_dict.values(), reverse=True)
            sorted_feature_dict = {k: v for k, v in zip(
                sorted_features, sorted_values)}
        else:
            raise Exception('unrecognized order.')
        features = list(sorted_feature_dict.keys())
        importances = list(sorted_feature_dict.values())
        # Limits number of features shown.
        features = features[:max_features]
        importances = importances[:max_features]
        # Arranges most important feature at top instead of bottom.
        features.reverse()
        importances.reverse()
        fig, ax = plt.subplots(**kwargs)
        ax.barh(range(len(features)), importances, align='center')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_title(title)
        plt.rc('font', size=fontsize)
        plt.show()
        plt.close()

# Yuri handles data plots.


class Yuri:

    def __init__(self, y_test=None, y_prob=None):
        self.classes = None

    def auc_score(self, y_test, y_prob):
        '''
        A wrapper on sklearn's roc_auc_score that makes it work even if the
        target is multi-categorical or has not been label encoded.

        The auc_score normally ranges between 0.5 and 1. Less than 0.5 makes
        the model worse than the baseline.

        The function assumes y_prob is ordered in ascending order for the
        target.
        '''
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_auc_score
        # Gets all unique classes.
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
        # Set tree model.
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
        # Sets hyperparameter.
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

    def plot_auc(self, x, auc_scores, lw=2, title=None, xlabel=None, labels=None, fontsize=10, **kwargs):
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
        plt.rc('font', size=fontsize)
        plt.show()
        plt.close()

    def plot_roc(self, y_test, y_prob, average='macro', mm=False, reverse_classes=False, lw=2, title=None, labels=None, fontsize=10, **kwargs):
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
        # Gets all unique classes.
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
        plt.rc('font', size=fontsize)
        plt.show()
        plt.close()
