from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Set, Literal

from minigrid.core.constants import COLOR_NAMES

@dataclass
class Product:
    coord: Tuple[int, int]
    color: str
    preference_reward: float
    irrational_preference_reward: float
    description: Optional[str] = None


@dataclass
class StoreLayout:
    width: int = 6
    height: int = 7
    # agent_initial_pos: Tuple[int, int] = (5,7)
    agent_initial_pos: Tuple[int, int] = (5,1)
    exit_pos: Tuple[int, int] = (2,1)
    # From top left, (column, row)
    shelves_positions: Set[Tuple[int, int]] = field(default_factory=lambda:set([
            (2, 0),
            (3, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3),
            (5, 0),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (1, 6),
            (2, 6),
            (3, 6),
        ]))  
    # List of products
    products_placement: List[Product] = field(
        default_factory=lambda: [
        Product(
            coord=(2,0), 
            color="blue", 
            preference_reward=0.25,
            irrational_preference_reward=0.2,
        ), 
        Product(
            coord=(5,3), 
            color="red", 
            preference_reward=0.25,
            irrational_preference_reward=0.3,
        ),
        Product(
            coord=(2,6), 
            color="yellow", 
            preference_reward=0.2,
            irrational_preference_reward=0.4,
        ),
        Product(
            coord=(0,2), 
            color="green", 
            preference_reward=0.3,
            irrational_preference_reward=0.1,
        ),]
    )
    

    def __post_init__(self):
        assert self.width>0 and self.height>0

        for (x,y) in self.shelves_positions:
            assert x<self.width and y<self.height
        
        for product in self.products_placement:
            assert product.coord in self.shelves_positions, "Products must be placed on shelves"


@dataclass
class StoreEnvSetup:
    max_steps: int = 128
    render_mode: str = "rgb_array"
    irrational_weight: float = .0
    time_penalty_per_step: float = -0.001
    time_out_penalty: float = .0
    exit_reward: float = .0
    sparse_reward: bool = False
    obs_type: Literal["state", "obs", "tabular"] = "state"
    layout: StoreLayout = field(default_factory=StoreLayout)

def interpolate_store_layout() -> StoreLayout:
    return StoreLayout(
        width=6,
        height=7,
        agent_initial_pos = (5,1),
        exit_pos = (2,1),
        shelves_positions = [
                (2, 0),
                (3, 0),
                # (5, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (2, 2),
                (2, 3),
                (3, 2),
                (3, 3),
                (5, 0),
                (5, 2),
                (5, 3),
                (5, 4),
                (5, 5),
                (1, 6),
                (2, 6),
                (3, 6),
        ],
        # List of products
        products_placement = [
            Product(
                coord=(2,0), 
                color="blue", 
                preference_reward=0.25,
                irrational_preference_reward=0.2,
            ), 
            Product(
                coord=(5,3), 
                color="red", 
                preference_reward=0.25,
                irrational_preference_reward=0.3,
            ),
            Product(
                coord=(2,6), 
                color="yellow", 
                preference_reward=0.2,
                irrational_preference_reward=0.4,
            ),
            Product(
                coord=(0,2), 
                color="green", 
                preference_reward=0.3,
                irrational_preference_reward=0.1,
            ),
        ]
    )

def make_env_to_show_interpolation(irrational_weight:float) -> StoreEnvSetup:
    return StoreEnvSetup(
        max_steps=128,
        irrational_weight=irrational_weight,
        time_penalty_per_step=-0.04,
        time_out_penalty=0.0,
        exit_reward=0.0,
        sparse_reward=False,
        layout=interpolate_store_layout()
    )

def make_env_always_buy_yellow(irrational_weight:float) -> StoreEnvSetup:
    layout = interpolate_store_layout()
    color_to_index = {prod.color:i for i,prod in enumerate(layout.products_placement)}
    
    ## Option 1
    layout.products_placement[color_to_index["yellow"]].coord = (0,2)
    layout.products_placement[color_to_index["blue"]].coord = (3,3)
    layout.products_placement[color_to_index["red"]].coord = (2,0)
    layout.products_placement[color_to_index["green"]].coord = (0,4)
    
    ## Option 2
    # layout.products_placement[color_to_index["yellow"]].coord = (0,3)

    ## Option 3
    # layout.products_placement[color_to_index["yellow"]].coord = (3,2)

    ## Option 4
    # layout.products_placement[color_to_index["yellow"]].coord = (5,4)
    # layout.products_placement[color_to_index["blue"]].coord = (3,3)
    # layout.products_placement[color_to_index["red"]].coord = (2,0)
    # layout.products_placement[color_to_index["green"]].coord = (0,4)

    return StoreEnvSetup(
        max_steps=128,
        irrational_weight=irrational_weight,
        time_penalty_per_step=-0.04,
        time_out_penalty=0.0,
        exit_reward=0.0,
        sparse_reward=False,
        layout=layout
    )