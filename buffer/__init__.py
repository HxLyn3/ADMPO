from .buffer import ReplayBuffer
from .buffer4seqsamp import ReplayBufferForSeqSampling

BUFFER = {
    "vanilla": ReplayBuffer,
    "seq-sample": ReplayBufferForSeqSampling
}
