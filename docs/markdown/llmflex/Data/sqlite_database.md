Module llmflex.Data.sqlite_database
===================================

Classes
-------

`SQLiteDatabase(db_dir: str)`
:   Initialising the SQLite database.
    
    Args:
        db_dir (str): Full directory of the database.

    ### Instance variables

    `db_dir: str`
    :   Full directory of the database.
        
        Returns:
            str: Full directory of the database.

    ### Methods

    `clear(self) ‑> None`
    :   Clear the entire database. Use it with caution.

    `create_table(self, sql: str, table_name: Optional[str] = None) ‑> str`
    :   Create a table from a SELECT query.
        
        Args:
            sql (str): SELECT query to create the table.
            table_name (Optional[str], optional): Name of the table. If None is given, a temp table name will be given. Defaults to None.
        
        Returns:
            str: Name of the created table.

    `df_to_table(self, df: pandas.core.frame.DataFrame, table_name: Optional[str] = None) ‑> str`
    :   Create a table from a Pandas dataframe.
        
        Args:
            df (pd.DataFrame): Pandas dataframe used to create the table.
            table_name (Optional[str], optional): Name of the table. If None is given, a temp table name will be given. Defaults to None.
        
        Returns:
            str: Name of the created table.

    `drop_table(self, table_name: str) ‑> None`
    :   Drop the table.
        
        Args:
            table_name (str): Name of table to drop.

    `execute(self, sql: str) ‑> None`
    :   Execute the given query.
        
        Args:
            sql (str): Query to execute.

    `list_tables(self) ‑> List[str]`
    :   List all the table names in the database.
        
        Returns:
            List[str]: List of all the table names in the database.

    `list_temp_tables(self) ‑> List[str]`
    :   List all the tables starting with 'temp_'.
        
        Returns:
            List[str]: List of all the tables starting with 'temp_'.

    `query(self, sql: str) ‑> pandas.core.frame.DataFrame`
    :   Execute the given SELECT query and return the result as a Pandas dataframe.
        
        Args:
            sql (str): SELECT query to execute.
        
        Returns:
            pd.DataFrame: Result of the query.