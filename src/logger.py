import logging 
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path, LOG_FILE)

# Configuración básica con archivo
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

#  Agregar consola además del archivo
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[ %(asctime)s ] %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

logging.getLogger().addHandler(console_handler)


'''if __name__=="__main__":
    logging.info("logging gas started")'''