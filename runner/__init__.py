from .online_trainer import ONTrainer
from .offline_trainer import OFFTrainer

TRAINER = {
    "online": ONTrainer,
    "offline": OFFTrainer
}
