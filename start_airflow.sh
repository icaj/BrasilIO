rm airflow.*
export AIRFLOW_HOME=`pwd`
cd $AIRFLOW_HOME
source venv/bin/activate
airflow standalone

