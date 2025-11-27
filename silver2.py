from pathlib import Path
import logging
from datetime import datetime

class BronzeToSilverPipeline:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.bronze_path = self.base_path / "bronze"
        self.silver_path = self.base_path / "silver"
        
        # Configura logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def validate_paths(self):
        """Valida se os paths existem"""
        if not self.bronze_path.exists():
            raise FileNotFoundError(f"Pasta bronze não encontrada: {self.bronze_path}")
        
        self.silver_path.mkdir(parents=True, exist_ok=True)
    
    def get_parquet_files(self):
        """Lista todos os arquivos Parquet na bronze"""
        return list(self.bronze_path.rglob("*.parquet"))
    
    def process_files(self):
        """Processa todos os arquivos"""
        self.validate_paths()
        
        parquet_files = self.get_parquet_files()
        self.logger.info(f"Encontrados {len(parquet_files)} arquivos Parquet")
        
        if not parquet_files:
            self.logger.warning("Nenhum arquivo Parquet encontrado")
            return None
        
        # Escolhe a estratégia baseada no número de arquivos
        if len(parquet_files) > 50:
            self.logger.info("Muitos arquivos, usando estratégia Spark")
            return self._process_with_spark()
        else:
            self.logger.info("Poucos arquivos, usando estratégia Pandas")
            return self._process_with_pandas()
    
    def _process_with_spark(self):
        """Processa com Spark para muitos arquivos"""
        try:
            from pyspark.sql import SparkSession
            
            spark = SparkSession.builder \
                .appName("BronzeToSilver") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            
            # Lê com merge de schema
            df = spark.read \
                .option("mergeSchema", "true") \
                .parquet(str(self.bronze_path))
            
            # Transformações
            df_transformed = self._apply_transformations_spark(df)
            
            # Salva
            df_transformed.write \
                .mode("overwrite") \
                .option("compression", "snappy") \
                .parquet(str(self.silver_path))
            
            self.logger.info(f"Dados salvos em: {self.silver_path}")
            
            spark.stop()
            return df_transformed
            
        except ImportError:
            self.logger.warning("Spark não disponível, usando fallback para Pandas")
            return self._process_with_pandas()
    
    def _process_with_pandas(self):
        """Processa com Pandas"""
        import pandas as pd
        
        all_dfs = []
        
        for file_path in self.get_parquet_files():
            try:
                df = pd.read_parquet(file_path)
                
                # Força tipos consistentes
                df = self._enforce_schema(df)
                all_dfs.append(df)
                
            except Exception as e:
                self.logger.error(f"Erro ao ler {file_path}: {e}")
        
        if not all_dfs:
            raise ValueError("Nenhum arquivo pôde ser lido")
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        transformed_df = self._apply_transformations_pandas(combined_df)
        
        # Salva
        transformed_df.to_parquet(
            self.silver_path / "dados_silver.parquet",
            index=False,
            compression='snappy'
        )
        
        self.logger.info(f"Dados salvos em: {self.silver_path}/dados_silver.parquet")
        return transformed_df
    
    def _enforce_schema(self, df):
        """Força schema consistente"""
        schema_rules = {
            'ano': 'int64',
            'mes': 'int32',
            'valor': 'float64'
        }
        
        for col, dtype in schema_rules.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    self.logger.warning(f"Erro ao converter {col} para {dtype}: {e}")
        
        return df
    
    def _apply_transformations_spark(self, df):
        """Transformações para Spark"""
        from pyspark.sql.functions import col, when
        
        # Exemplo de transformações
        df_transformed = df
        
        if 'ano' in df.columns:
            df_transformed = df_transformed.filter(col("ano").isNotNull())
        
        return df_transformed
    
    def _apply_transformations_pandas(self, df):
        """Transformações para Pandas"""
        df_transformed = df.copy()
        
        # Remove linhas com anos inválidos
        if 'ano' in df_transformed.columns:
            df_transformed = df_transformed[df_transformed['ano'].notna()]
            df_transformed = df_transformed[df_transformed['ano'] > 0]
        
        # Remove duplicatas
        df_transformed = df_transformed.drop_duplicates()
        
        return df_transformed

# USO
if __name__ == "__main__":
    pipeline = BronzeToSilverPipeline("dataset")
    result = pipeline.process_files()