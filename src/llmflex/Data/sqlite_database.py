import os
import pandas as pd
from typing import List, Optional

class SQLiteDatabase:

    def __init__(self, db_dir: str) -> None:
        """Initialising the SQLite database.

        Args:
            db_dir (str): Full directory of the database.
        """
        self._db_dir = os.path.abspath(db_dir)
        os.makedirs(os.path.dirname(self.db_dir), exist_ok=True)

    @property
    def db_dir(self) -> str:
        """Full directory of the database.

        Returns:
            str: Full directory of the database.
        """
        return self._db_dir
    
    def _connect(self) -> None:
        """Connect to the database.
        """
        import sqlite3 as sl3
        self._conn = sl3.connect(self.db_dir)

    def _disconnect(self) -> None:
        """Disconnect the database.
        """
        self._conn.close()

    def execute(self, sql: str) -> None:
        """Execute the given query.

        Args:
            sql (str): Query to execute.
        """
        sql = sql + ';' if not sql.endswith(';') else sql
        self._connect()
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql)
            self._conn.commit()
            cursor.close()
        except Exception as e:
            self._disconnect()
            raise e
        self._disconnect()

    def query(self, sql: str) -> pd.DataFrame:
        """Execute the given SELECT query and return the result as a Pandas dataframe.

        Args:
            sql (str): SELECT query to execute.

        Returns:
            pd.DataFrame: Result of the query.
        """
        sql = sql + ';' if not sql.endswith(';') else sql
        self._connect()
        try:
            df = pd.read_sql(sql, con=self._conn)
        except Exception as e:
            self._disconnect()
            raise e
        self._disconnect()
        return df
    
    def list_tables(self) -> List[str]:
        """List all the table names in the database.

        Returns:
            List[str]: List of all the table names in the database.
        """
        tables = self.query("SELECT name FROM sqlite_schema WHERE type='table'")
        return tables['name'].values.tolist()
    
    def list_temp_tables(self) -> List[str]:
        """List all the tables starting with 'temp_'.

        Returns:
            List[str]: List of all the tables starting with 'temp_'.
        """
        return list(filter(lambda x: x.startswith('temp_'), self.list_tables()))
    
    def _new_temp_table_name(self) -> str:
        """Generate a new temp table name given the existing temp tables.

        Returns:
            str: New temp table name.
        """
        temp_tables = self.list_temp_tables()
        indice = list(map(lambda x: int(x.removeprefix('temp_')), temp_tables))
        if len(indice) == 0:
            return 'temp_0'
        else:
            return f'temp_{max(indice) + 1}'
        
    def drop_table(self, table_name: str) -> None:
        """Drop the table.

        Args:
            table_name (str): Name of table to drop.
        """
        if table_name not in self.list_tables():
            print(f'"{table_name}" does not exist.')
        else:
            self.execute(f'DROP TABLE {table_name};')
        
    def create_table(self, sql: str, table_name: Optional[str] = None) -> str:
        """Create a table from a SELECT query.

        Args:
            sql (str): SELECT query to create the table.
            table_name (Optional[str], optional): Name of the table. If None is given, a temp table name will be given. Defaults to None.

        Returns:
            str: Name of the created table.
        """
        table_name = self._new_temp_table_name() if table_name is None else table_name
        sql = sql.removesuffix(';')
        self.execute(f'DROP TABLE IF EXISTS {table_name};')
        self.execute(f'CREATE TABLE {table_name} AS ({sql});')
        return table_name
    
    def df_to_table(self, df: pd.DataFrame, table_name: Optional[str] = None) -> str:
        """Create a table from a Pandas dataframe.

        Args:
            df (pd.DataFrame): Pandas dataframe used to create the table.
            table_name (Optional[str], optional): Name of the table. If None is given, a temp table name will be given. Defaults to None.

        Returns:
            str: Name of the created table.
        """
        table_name = self._new_temp_table_name() if table_name is None else table_name
        self._connect()
        try: 
            df.to_sql(table_name, con=self._conn, if_exists='replace', index=False)
        except Exception as e:
            self._disconnect()
            raise e
        self._disconnect()
        return table_name
    
    def clear(self) -> None:
        """Clear the entire database. Use it with caution.
        """
        os.remove(self.db_dir)
        self._connect()
        self._disconnect()
        



