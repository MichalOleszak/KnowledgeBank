# Import the DAG object
from airflow.models import DAG

# Define the default_args dictionary
default_args = {
  'owner': 'dsmith',
  'start_date': datetime(2020, 1, 14),
  'retries': 2
}

# Instantiate the DAG object
etl_dag = DAG("example_etl", default_args=default_args)

# List DAGs
airflow list_dags

# Start webserver
airflow webserver -p 9090

# Operators
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import Pythonperator
from airflow.operators.email_operator import EmailOperator

cleanup_task = BashOperator(
    task_id="cleanup_task",
    # Define the bash_command
    bash_command="cleanup.sh",
    # Add the task to the dag
    dag=etl_dag
)

def sleep(secs):
  time.sleep(secs)

sleep_task = PythonOperator(
  task_id="sleep",
  python_callable=sleep,
  op_kwargs={"secs": 5},
  dag=etl_dag
)

email_task = EmailOperator(
  task_id="email",
  to="a@b.com",
  subject="abc",
  html_content="abc",
  files="myfile.csv",
  dag=etl_dag
)

# Tasks - instances of operators
# Task dependencies define order ot task completion:
# - upstream >> tasks are executed before
# - downstream << 

# run cleanup before dump before email:
cleanup_task >> dump_task >> email_task


# Airflow scheduling == running DAGs
default_args = {
  'owner': 'Engineering',
  'start_date': datetime(2019, 11, 1),
  'email': ['airflowresults@datacamp.com'],
  'email_on_failure': False,
  'email_on_retry': False,
  'retries': 3,
  'retry_delay': timedelta(minutes=20)
}
dag = DAG('update_dataflows', default_args=default_args, schedule_interval='30 12 * * 3')


# Sensors == operators waiting for a condition to be true
from airflow.contrib.sensors.file_sensor import FileSensor

file_sensor_task = FileSensor(
  task_id="wait_for_file_to_exist",
  filepath="avc.csv",
  poke_interaval=300,  # check every 5 mins
  dag=example_dag
)

generate_file >> file_sensor_task >> do_stuff_using_file


# Executors == components actually running tasks
# determine executor type
cat airflow/airflow.cfg | grep "executor = "
airflow list_dags  # --> info in output


# SLA = service level agreement (amount of time a task or dag should require to run)
# can be set in operators or via default args when creating a dag
cleanup_task = BashOperator(
    task_id="cleanup_task",
    bash_command="cleanup.sh",
    sla=datetime.timedelta(seconds=30),
    dag=etl_dag
)

default_args = {
  'start_date': datetime(2020, 2, 20),
  'sla': datetime.timedelta(minutes=30)
}
test_dag = DAG('test_workflow', default_args=default_args, schedule_interval='@None')


# Reporting - receive a report from Airflow when tasks complete
email_report = EmailOperator(
        task_id='email_report',
        to='airflow@abc.com',
        subject='Airflow Monthly Report',
        html_content="""Attached is your monthly workflow report - please refer to it for more detail""",
        files=["monthly_report.pdf"],
        dag=report_dag
)
email_report << generate_report

# Send email on task success and failure
default_args={
    'email': ["airflowalerts@abc.com", "airflowadmin@abc.com"],
    'email_on_failure': True,
    'email_on_success': True
}
report_dag = DAG(
    dag_id = 'execute_report',
    schedule_interval = "0 0 * * *",
    default_args=default_args
)


# Templates
default_args = {
  'start_date': datetime(2020, 4, 15),
}
cleandata_dag = DAG('cleandata',
                    default_args=default_args,
                    schedule_interval='@daily')
templated_command = """
  bash cleandata.sh {{ ds_nodash }} {{ params.filename }}
"""
clean_task = BashOperator(task_id='cleandata_task',
                          bash_command=templated_command,
                          params={'filename': "salesdata.txt"},
                          dag=cleandata_dag)

# Airflow built-in variables:
# {{ ds }} - execution date as YYYY-MM-DD
# {{ ds_nodash }} - execution date as YYYMMDD
# {{ prev_ds }}, {{ prev_ds_nodash }} - previous execution date
# {{ dag }} - DAG object
# {{ conf }} - configuration object
# {{ macros.datetime }} - python datetime object
# {{ macros.timedelta }} - python timedelta object
# {{macros.ds_add('2020-04-15', 5) }} - function, returns '2020-04-20'

# Run the same operator for many files collected in filenames_list
templated_command = """
  <% for filename in params.filenames %>
  bash cleandata.sh {{ ds_nodash }} {{ filename }};
  <% endfor %>
"""
clean_task = BashOperator(task_id='cleandata_task',
                          bash_command=templated_command,
                          params={'filenames': filenames_list},
                          dag=cleandata_dag)


# Branching - provides conditional logic
# BranchPythonOperator takes a function that accepts kwargs and returns the ids of tasks to run
def branch_test(**kwargs):
  if int(kwargs["ds_nodash"]) % 2 == 0:
    return "even_day_task"
 else:
    return "odd_day_task"

branch_task = BranchPythonOperator(
  task_id="branch_task",
  provide_context=True,  # provide access to the runtime variables and macros to the function (kwargs)
  python_callable=branch_test,
)

start_task >> branch_task >> even_day_task
branch_task >> odd_day_task

# Running
airflow run <dag_id> <task_id> <execution_date>
airflow trigger_dag -e <execution_date> <dag_id>


