from .hopper import StaticFns as HopperStaticFns
from .swimmer import StaticFns as SwimmerStaticFns
from .walker2d import StaticFns as Walker2dStaticFns
from .halfcheetah import StaticFns as HalfcheetahStaticFns
from .inverted_pendulum import StaticFns as InvertedPendulumFns
from .ant_truncated_obs import StaticFns as AntTruncatedObsStaticFns
from .humanoid_truncated_obs import StaticFns as HumanoidTruncatedObsStaticFns

from .neorl_hopper import StaticFns as NeoRLHopperStaticFns
from .neorl_walker2d import StaticFns as NeoRLWalker2dStaticFns
from .neorl_halfcheetah import StaticFns as NeoRLHalfcheetahStaticFns

from .pen import StaticFns as PenStaticFns
from .door import StaticFns as DoorStaticFns
from .hammer import StaticFns as HammerStaticFns

from .antmaze_umaze import StaticFns as AntMazeUmazeStaticFns
from .antmaze_medium import StaticFns as AntMazeMediumStaticFns
from .antmaze_large import StaticFns as AntMazeLargeStaticFns

STATICFUNC = {
    "hopper": HopperStaticFns,
    "swimmer": SwimmerStaticFns,
    "walker2d": Walker2dStaticFns,
    "halfcheetah": HalfcheetahStaticFns,
    "invertedpendulum": InvertedPendulumFns,
    "anttruncatedobs": AntTruncatedObsStaticFns,
    "humanoidtruncatedobs": HumanoidTruncatedObsStaticFns,

    "neorl-hopper": NeoRLHopperStaticFns,
    "neorl-walker2d": NeoRLWalker2dStaticFns,
    "neorl-halfcheetah": NeoRLHalfcheetahStaticFns,

    "pen": PenStaticFns,
    "door": DoorStaticFns,
    "hammer": HammerStaticFns,

    "antmaze-umaze": AntMazeUmazeStaticFns,
    "antmaze-medium": AntMazeMediumStaticFns,
    "antmaze-large": AntMazeLargeStaticFns
}
