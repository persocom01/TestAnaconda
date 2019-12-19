import sqlite3
import click
from flask import current_app, g
from flask.cli import with_appcontext


def get_db():
    # g is a special request object. It stores request data so that multiple
    # functions may call on it without initiating a new request.
    if 'db' not in g:
        # Establishes a connection to the file pointed at by the DATABASE
        # config key.
        g.db = sqlite3.connect(
            # current_app is special object pointing to the application
            # handling the request.
            # detect_types=sqlite3.PARSE_DECLTYPES tries to convert sqlite
            # datatypes to python when data is drawn from the database. One
            # effect is trying to convert strings to datetime objects and
            # returning an error when it is unable to do so.
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        # Makes the database return a dictionary-like object instead of a
        # tuple.
        g.db.row_factory = sqlite3.Row

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()


def init_db():
    db = get_db()
    # app.open_resource(path_relative_to_app, mode='rb')
    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))


# Defines a cmd command.
@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')


def init_app(app):
    # app.teardown_appcontext(func) registers a function to be called when the
    # app context ends.
    app.teardown_appcontext(close_db)
    # app.cli.add_command(cmd, name=None) adds a cmd command. The name of the
    # command is used if name is not defined.
    app.cli.add_command(init_db_command)
