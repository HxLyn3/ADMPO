from .actor import ProbActor, DeterActor
from .critic import Critic

ACTOR = {
    "prob": ProbActor,
    "deter": DeterActor
}

CRITIC = {
    "q": Critic,
    "v": None
}
