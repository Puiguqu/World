# Copy and run the entire cell at once
import random
import math
import time
from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional, Set
from IPython.display import clear_output

# ===== BASE CLASSES =====

class Vector3:
    """A simple 3D vector class for position, velocity, and acceleration."""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
    
    def distance_to(self, other) -> float:
        """Calculate Euclidean distance to another vector in 3D space."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def horizontal_distance_to(self, other) -> float:
        """Calculate horizontal distance ignoring height (y-axis)."""
        return np.sqrt((self.x - other.x)**2 + (self.z - other.z)**2)


class TimeSystem:
    """Manages the time in the simulation with a 24-hour cycle."""
    
    def __init__(self):
        self.hour = 7  # Starting at 7 AM
        self.minute = 0
        self.day = 1
        self.minute_increment = 10  # Each tick advances time by 10 minutes
        
        # Time settings
        self.sunrise = 7  # 7 AM
        self.sunset = 20  # 8 PM
        self.work_start = 9  # 9 AM
        self.work_end = 17  # 5 PM
    
    def update(self):
        """Advance time by one increment."""
        self.minute += self.minute_increment
        if self.minute >= 60:
            self.hour += 1
            self.minute = 0
            
        if self.hour >= 24:
            self.hour = 0
            self.day += 1
    
    def is_daytime(self) -> bool:
        """Check if it's currently daytime (sun is up)."""
        return self.sunrise <= self.hour < self.sunset
    
    def is_work_hours(self) -> bool:
        """Check if it's currently working hours."""
        return self.work_start <= self.hour < self.work_end and self.is_weekday()
    
    def is_weekday(self) -> bool:
        """Check if it's a weekday (not weekend)."""
        # Assuming day 1 is Monday, so days 6 and 7 are weekend
        return (self.day % 7) not in [6, 0]  # 0 is Sunday in this system
    
    def get_time_string(self) -> str:
        """Get a formatted time string."""
        am_pm = "AM" if self.hour < 12 else "PM"
        display_hour = self.hour if self.hour <= 12 else self.hour - 12
        if display_hour == 0:
            display_hour = 12
        return f"Day {self.day}, {display_hour}:{self.minute:02d} {am_pm}"


# ===== ENUMERATIONS =====

class BuildingType(Enum):
    """Types of buildings in the city."""
    RESIDENTIAL = auto()
    COMMERCIAL = auto()
    INDUSTRIAL = auto()
    GOVERNMENT = auto()
    POLICE_STATION = auto()
    EMPTY = auto()


class NeedType(Enum):
    """Basic needs that citizens try to fulfill."""
    SLEEP = auto()
    FOOD = auto()
    INCOME = auto()
    RECREATION = auto()
    SAFETY = auto()


class CitizenPersonality(Enum):
    """Different personality types that influence behavior."""
    LAWFUL = auto()    # Follows rules, unlikely to commit crimes
    NEUTRAL = auto()   # Average citizen, might break minor laws
    CRIMINAL = auto()  # More likely to engage in criminal activities


class CitizenState(Enum):
    """Possible states that a citizen can be in."""
    IDLE = auto()
    WORKING = auto()
    COMMUTING = auto()
    SHOPPING = auto()
    SLEEPING = auto()
    RECREATING = auto()
    COMMITTING_CRIME = auto()
    ARRESTED = auto()
    HIDING = auto()
    CLIMBING = auto()  # New state for vertical movement


class CrimeType(Enum):
    """Types of crimes that can be committed in the simulation."""
    THEFT = auto()
    VANDALISM = auto()
    ASSAULT = auto()
    BURGLARY = auto()
    SABOTAGE = auto()  # Organized attack on infrastructure


# ===== PHYSICS SYSTEM =====

class Entity:
    """An entity in the physics simulation with physical properties."""
    def __init__(self, x=0.0, y=0.0, z=0.0, mass=1.0):
        self.position = Vector3(x, y, z)
        self.velocity = Vector3(0, 0, 0)
        self.acceleration = Vector3(0, 0, 0)
        self.mass = mass
        self.restitution = 0.7  # Bounciness factor (0 = no bounce, 1 = perfect bounce)
        self.grounded = False
        self.radius = 0.5  # For collision detection with buildings
        self.height = 1.8  # Default human height
        self.is_building = False  # Flag for collision detection


class PhysicsSimulation:
    """Physics simulation with gravity and basic collision handling."""
    def __init__(self):
        # Earth's gravity in m/s²
        self.gravity = 9.81
        
        # Simulation settings
        self.time_step = 0.016  # 60fps equivalent in seconds
        self.ground_level = 0.0  # y-coordinate of ground level
        
        # Collection of all entities in the simulation
        self.entities = []
        self.buildings = []
    
    def add_entity(self, entity):
        """Add an entity to the simulation."""
        # Ensure entity is at or above ground level
        if entity.position.y < self.ground_level:
            entity.position.y = self.ground_level
        
        self.entities.append(entity)
        if entity.is_building:
            self.buildings.append(entity)
        return entity
    
    def remove_entity(self, entity):
        """Remove an entity from the simulation."""
        if entity in self.entities:
            self.entities.remove(entity)
        if entity in self.buildings:
            self.buildings.remove(entity)
    
    def apply_force(self, entity, force_x, force_y, force_z):
        """Apply a force to an entity."""
        # F = ma, so a = F/m
        entity.acceleration.x += force_x / entity.mass
        entity.acceleration.y += force_y / entity.mass
        entity.acceleration.z += force_z / entity.mass
    
    def is_grounded(self, entity):
        """Check if an entity is on the ground or on top of a building."""
        # Check if on ground
        if abs(entity.position.y - self.ground_level) < 0.01 and entity.velocity.y <= 0:
            return True
            
        # Check if on top of any building
        for building in self.buildings:
            if (abs(entity.position.x - building.position.x) <= building.width/2 and
                abs(entity.position.z - building.position.z) <= building.depth/2 and
                abs(entity.position.y - building.height) < 0.1 and
                entity.velocity.y <= 0):
                return True
                
        return False
    
    def update(self):
        """Apply forces, update positions, and handle collisions."""
        for entity in self.entities:
            if entity.is_building:
                # Buildings are static, no need to update physics
                continue
                
            # Apply gravity if not grounded
            if not entity.grounded:
                self.apply_force(entity, 0, -self.gravity * entity.mass, 0)
            
            # Update velocity based on acceleration
            entity.velocity.x += entity.acceleration.x * self.time_step
            entity.velocity.y += entity.acceleration.y * self.time_step
            entity.velocity.z += entity.acceleration.z * self.time_step
            
            # Reset acceleration
            entity.acceleration.x = 0
            entity.acceleration.y = 0
            entity.acceleration.z = 0
            
            # Apply velocity limits (optional)
            max_horizontal_speed = getattr(entity, 'max_speed', 10.0)
            horizontal_speed = np.sqrt(entity.velocity.x**2 + entity.velocity.z**2)
            if horizontal_speed > max_horizontal_speed:
                scale = max_horizontal_speed / horizontal_speed
                entity.velocity.x *= scale
                entity.velocity.z *= scale
                
            # Update position based on velocity
            entity.position.x += entity.velocity.x * self.time_step
            entity.position.y += entity.velocity.y * self.time_step
            entity.position.z += entity.velocity.z * self.time_step
            
            # Check for ground collision
            if entity.position.y <= self.ground_level:
                entity.position.y = self.ground_level
                
                # Bounce if moving downward
                if entity.velocity.y < 0:
                    entity.velocity.y = -entity.velocity.y * entity.restitution
                    
                    # Apply friction to horizontal movement
                    entity.velocity.x *= 0.95
                    entity.velocity.z *= 0.95
                    
                    # Stop very small bounces
                    if abs(entity.velocity.y) < 0.1:
                        entity.velocity.y = 0
                
                entity.grounded = True
            else:
                entity.grounded = self.is_grounded(entity)


# ===== BUILDING CLASS =====

class Building(Entity):
    """A building in the 3D city simulation."""
    def __init__(self, x, y, z, width, height, depth, building_type):
        super().__init__(x, y, z, mass=1000.0)  # Buildings have high mass
        self.width = width
        self.height = height
        self.depth = depth
        self.type = building_type
        self.is_building = True
        
        # Building properties
        self.floors = max(1, int(height / 3))  # Assuming 3 meters per floor
        self.occupants = []
        self.wealth = np.random.uniform(0.5, 1.5)  # Building quality/value
        
        # Job-related properties
        if building_type in [BuildingType.COMMERCIAL, BuildingType.INDUSTRIAL, 
                           BuildingType.GOVERNMENT, BuildingType.POLICE_STATION]:
            self.jobs = self.floors * np.random.randint(3, 10)  # Jobs per floor
            
            if building_type == BuildingType.COMMERCIAL:
                self.wage = np.random.uniform(8, 25)
            elif building_type == BuildingType.INDUSTRIAL:
                self.wage = np.random.uniform(10, 20)
            elif building_type == BuildingType.GOVERNMENT:
                self.wage = np.random.uniform(15, 30)
            elif building_type == BuildingType.POLICE_STATION:
                self.wage = np.random.uniform(18, 28)
        else:
            self.jobs = 0
            self.wage = 0
            
        # Track job openings and employees
        self.employees = []
        self.job_openings = self.jobs
    
    def add_occupant(self, citizen):
        """Add a citizen as an occupant of this building."""
        self.occupants.append(citizen)
    
    def remove_occupant(self, citizen):
        """Remove a citizen from this building's occupants."""
        if citizen in self.occupants:
            self.occupants.remove(citizen)
    
    def add_employee(self, citizen):
        """Hire a citizen as an employee."""
        if self.job_openings > 0:
            self.employees.append(citizen)
            self.job_openings -= 1
            return True
        return False
    
    def remove_employee(self, citizen):
        """Remove a citizen from employment."""
        if citizen in self.employees:
            self.employees.remove(citizen)
            self.job_openings += 1
    
    def get_entrance_position(self):
        """Get the position of the building entrance."""
        # Simplified: entrance is at the center of one side of the building
        return Vector3(
            self.position.x - self.width/2,  # Front entrance on -X side
            0,  # Ground level
            self.position.z
        )
    
    def get_random_position_inside(self, floor=None):
        """Get a random position inside the building, optionally on a specific floor."""
        if floor is None:
            floor = np.random.randint(0, self.floors)
        else:
            floor = min(max(0, floor), self.floors - 1)
            
        floor_height = 3.0  # 3 meters per floor
            
        return Vector3(
            self.position.x + np.random.uniform(-self.width/2 + 0.5, self.width/2 - 0.5),
            floor_height * (floor + 0.5),  # Middle of the floor
            self.position.z + np.random.uniform(-self.depth/2 + 0.5, self.depth/2 - 0.5)
        )
    
    def contains_point(self, point):
        """Check if a 3D point is inside the building."""
        return (abs(point.x - self.position.x) <= self.width/2 and
                point.y >= 0 and point.y <= self.height and
                abs(point.z - self.position.z) <= self.depth/2)


# ===== CITIZEN CLASS =====

class Citizen(Entity):
    """A citizen in the 3D city simulation with needs, personality, and behavior."""
    
    def __init__(self, city, x: float, y: float, z: float):
        super().__init__(x, y, z, mass=70.0)  # Average human mass
        self.city = city
        self.target_position = None
        self.max_speed = 1.5  # Maximum walking speed in m/s
        self.climb_speed = 0.5  # Vertical movement speed in m/s (climbing stairs)
        self.home = None
        self.home_floor = 0  # Floor number in residential building
        self.workplace = None
        self.workplace_floor = 0  # Floor number in workplace
        self.state = CitizenState.IDLE
        self.restitution = 0.2  # Less bouncy than default
        
        # Movement mechanics
        self.jump_force = 5.0  # Jump force for vertical movement
        self.jump_cooldown = 0
        
        # Citizen properties
        self.name = f"Citizen-{id(self) % 10000}"
        self.age = random.randint(18, 65)
        self.money = random.uniform(0, 50000000)
        self.personality = np.random.choice(
            list(CitizenPersonality),
            p=[0.7, 0.25, 0.05]  # 70% lawful, 25% neutral, 5% criminal
        )
        
        # Needs system (0-100 scale)
        self.needs = {
            NeedType.SLEEP: random.uniform(40, 80),
            NeedType.FOOD: random.uniform(40, 80),
            NeedType.INCOME: random.uniform(40, 80),
            NeedType.RECREATION: random.uniform(40, 80),
            NeedType.SAFETY: random.uniform(40, 80)
        }
        
        # Crime history
        self.criminal_record = []
        self.crimes_committed = 0
        self.arrest_time = 0  # Remaining time if arrested
        self.current_crime = None
    
    def update(self, time_system):
        """Update citizen state based on needs, time, and environment."""
        # Skip update if arrested
        if self.state == CitizenState.ARRESTED:
            self.arrest_time -= 1
            if self.arrest_time <= 0:
                self.state = CitizenState.IDLE
                if self.city.police_stations:
                    station = self.city.police_stations[0]
                    self.position.x = station.get_entrance_position().x
                    self.position.y = 0
                    self.position.z = station.get_entrance_position().z
            return
        
        # Reduce jump cooldown if active
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
        
        # Update needs based on time
        self._update_needs(time_system)
        
        # State machine for citizen behavior
        if self.state == CitizenState.COMMUTING:
            self._move_to_target()
        elif self.state == CitizenState.CLIMBING:
            self._climb_to_target()
        elif self.state == CitizenState.IDLE:
            self._decide_next_action(time_system)
        elif self.state == CitizenState.SLEEPING:
            if not time_system.is_daytime() or self.needs[NeedType.SLEEP] < 30:
                # Continue sleeping
                self.needs[NeedType.SLEEP] += 2.0
            else:
                # Wake up
                self.state = CitizenState.IDLE
        elif self.state == CitizenState.WORKING:
            if time_system.is_work_hours():
                # Work and earn money
                self.money += self.workplace.wage * (time_system.minute_increment / 60)
                self.needs[NeedType.INCOME] += 1.0
                self.needs[NeedType.RECREATION] -= 0.5
            else:
                # End workday
                self.state = CitizenState.IDLE
                # Go to building exit first
                exit_pos = self.workplace.get_entrance_position()
                if self.position.y > 0.5:
                    # Need to go down to exit
                    self._set_target(exit_pos.x, 0, exit_pos.z)
                    self.state = CitizenState.CLIMBING
                    self.next_state = CitizenState.COMMUTING
                    self.next_target = self.home.get_entrance_position()
                else:
                    # Already at ground level
                    self._set_target(exit_pos.x, 0, exit_pos.z)
                    self.state = CitizenState.COMMUTING
                    self.next_target = self.home.get_entrance_position()
        elif self.state == CitizenState.COMMITTING_CRIME:
            self._commit_crime()
            
        # Check if should consider crime
        if (self.personality == CitizenPersonality.CRIMINAL or 
            (self.personality == CitizenPersonality.NEUTRAL and 
             any(value < 20 for value in self.needs.values()))):
            self._consider_crime(time_system)
    
    def _update_needs(self, time_system):
        """Update citizen needs based on time and activities."""
        # Basic decay rates
        self.needs[NeedType.SLEEP] -= 0.3 * (time_system.minute_increment / 60)
        self.needs[NeedType.FOOD] -= 0.2 * (time_system.minute_increment / 60)
        self.needs[NeedType.INCOME] -= 0.1 * (time_system.minute_increment / 60)
        self.needs[NeedType.RECREATION] -= 0.15 * (time_system.minute_increment / 60)
        
        # Safety need depends on nearby police and crime
        police_presence = self.city.get_police_presence(self.position)
        self.needs[NeedType.SAFETY] = min(100, self.needs[NeedType.SAFETY] + 
                                         (police_presence * 0.5 - 0.1) * 
                                         (time_system.minute_increment / 60))
        
        # Clamp needs to valid range
        for need_type in self.needs:
            self.needs[need_type] = max(0, min(100, self.needs[need_type]))
    
    def _decide_next_action(self, time_system):
        """Decide what to do next based on needs and time."""
        # Sleep at night or when sleep need is low
        if not time_system.is_daytime() or self.needs[NeedType.SLEEP] < 20:
            if self.home:
                if self.position.horizontal_distance_to(self.home.position) < 3.0:
                    if abs(self.position.y - (self.home_floor * 3.0)) < 1.0:
                        # Already in home at correct floor, go to sleep
                        self.state = CitizenState.SLEEPING
                    else:
                        # Need to go to correct floor
                        home_pos = self.home.get_random_position_inside(floor=self.home_floor)
                        self._set_target(home_pos.x, home_pos.y, home_pos.z)
                        self.state = CitizenState.CLIMBING
                else:
                    # Head to home building entrance first
                    entrance = self.home.get_entrance_position()
                    self._set_target(entrance.x, entrance.y, entrance.z)
                    self.state = CitizenState.COMMUTING
                    self.next_target = self.home.get_random_position_inside(floor=self.home_floor)
                    self.next_state = CitizenState.CLIMBING
            return
        
        # Go to work during work hours
        if time_system.is_work_hours() and self.workplace:
            work_entrance = self.workplace.get_entrance_position()
            work_position = self.workplace.get_random_position_inside(floor=self.workplace_floor)
            
            if self.position.horizontal_distance_to(self.workplace.position) < 3.0:
                if abs(self.position.y - work_position.y) < 1.0:
                    # Already at workplace at right floor
                    self.state = CitizenState.WORKING
                else:
                    # Need to go to correct floor
                    self._set_target(work_position.x, work_position.y, work_position.z)
                    self.state = CitizenState.CLIMBING
            else:
                # Head to workplace building
                self._set_target(work_entrance.x, work_entrance.y, work_entrance.z)
                self.state = CitizenState.COMMUTING
                self.next_target = work_position
                self.next_state = CitizenState.CLIMBING
            return
        
        # Handle critical needs
        critical_needs = sorted(
            [(need_type, value) for need_type, value in self.needs.items() if value < 30],
            key=lambda x: x[1]
        )
        
        if critical_needs:
            critical_need = critical_needs[0][0]
            if critical_need == NeedType.FOOD and self.money > 10:
                # Find commercial building for food
                target = self.city.find_nearest_building(
                    self.position, BuildingType.COMMERCIAL
                )
                if target:
                    entrance = target.get_entrance_position()
                    self._set_target(entrance.x, entrance.y, entrance.z)
                    self.state = CitizenState.COMMUTING
            elif critical_need == NeedType.RECREATION:
                # Find a place to recreate
                target = self.city.find_random_building_of_type(
                    [BuildingType.COMMERCIAL]
                )
                if target:
                    entrance = target.get_entrance_position()
                    self._set_target(entrance.x, entrance.y, entrance.z)
                    self.state = CitizenState.COMMUTING
        
        # Just wander around if nothing else to do
        if self.state == CitizenState.IDLE:
            target_x = random.uniform(0, self.city.width - 1)
            target_z = random.uniform(0, self.city.depth - 1)
            self._set_target(target_x, 0, target_z)  # Stay at ground level for wandering
            self.state = CitizenState.COMMUTING
    
    def _move_to_target(self):
        """Move citizen toward target position in 3D space."""
        if not self.target_position:
            self.state = CitizenState.IDLE
            return
        
        # Calculate horizontal direction
        dx = self.target_position.x - self.position.x
        dz = self.target_position.z - self.position.z
        horizontal_distance = np.sqrt(dx**2 + dz**2)
        
        if horizontal_distance < 0.3:
            # Reached target horizontally
            self.position.x = self.target_position.x
            self.position.z = self.target_position.z
            
            # If there's a height difference, start climbing
            if abs(self.position.y - self.target_position.y) > 0.3:
                self.state = CitizenState.CLIMBING
                return
            else:
                self.position.y = self.target_position.y
                self.target_position = None
                
                # Check if there's a next state or target
                if hasattr(self, 'next_state') and hasattr(self, 'next_target'):
                    self.state = self.next_state
                    self._set_target(self.next_target.x, self.next_target.y, self.next_target.z)
                    delattr(self, 'next_state')
                    delattr(self, 'next_target')
                else:
                    self.state = CitizenState.IDLE
                return
        
        # Move toward target horizontally
        move_distance = min(self.max_speed * self.city.physics.time_step, horizontal_distance)
        
        # Calculate normalized horizontal direction vector
        normalized_dx = dx / horizontal_distance
        normalized_dz = dz / horizontal_distance
        
        # Apply movement
        self.velocity.x = normalized_dx * self.max_speed
        self.velocity.z = normalized_dz * self.max_speed
    
    def _climb_to_target(self):
        """Handle vertical movement to target height."""
        if not self.target_position:
            self.state = CitizenState.IDLE
            return
        
        # Check if we're horizontally at the target
        horizontal_distance = np.sqrt(
            (self.target_position.x - self.position.x)**2 + 
            (self.target_position.z - self.position.z)**2
        )
        
        if horizontal_distance > 0.3:
            # Not at target horizontally, switch to walking
            self.state = CitizenState.COMMUTING
            return
        
        # Calculate vertical direction
        dy = self.target_position.y - self.position.y
        
        if abs(dy) < 0.3:
            # Reached target vertically
            self.position.y = self.target_position.y
            self.velocity.y = 0
            self.target_position = None
            
            # Check if there's a next state
            if hasattr(self, 'next_state'):
                self.state = self.next_state
                delattr(self, 'next_state')
                
                # Check if there's a next target too
                if hasattr(self, 'next_target'):
                    self._set_target(self.next_target.x, self.next_target.y, self.next_target.z)
                    delattr(self, 'next_target')
            else:
                self.state = CitizenState.IDLE
            return
        
        # Move up or down at climb speed
        if dy > 0:
            # Climbing up
            self.velocity.y = self.climb_speed
        else:
            # Climbing down
            self.velocity.y = -self.climb_speed
    
    def jump(self):
        """Make the citizen jump if on the ground and not in cooldown."""
        if self.grounded and self.jump_cooldown <= 0:
            self.velocity.y = self.jump_force
            self.jump_cooldown = 20  # Frames before can jump again
            return True
        return False
    
    def _set_target(self, x, y, z):
        """Set a new movement target in 3D space."""
        self.target_position = Vector3(x, y, z)
    
    def _consider_crime(self, time_system):
        """Decide whether to commit a crime based on personality and needs."""
        # Skip if already committing a crime
        if self.state == CitizenState.COMMITTING_CRIME:
            return
        
        # Crime chance based on personality, needs, time of day
        crime_chance = 0.0
        
        if self.personality == CitizenPersonality.CRIMINAL:
            crime_chance = 0.01  # Base chance for criminals
        elif self.personality == CitizenPersonality.NEUTRAL:
            crime_chance = 0.001  # Much lower for neutral citizens
        else:
            crime_chance = 0.0001  # Nearly zero for lawful citizens
        
        # Modify based on needs
        if self.needs[NeedType.INCOME] < 20:
            crime_chance *= 3
        if self.needs[NeedType.FOOD] < 20:
            crime_chance *= 2
            
        # Higher chance at night
        if not time_system.is_daytime():
            crime_chance *= 2
            
        # Lower chance with police nearby
        police_presence = self.city.get_police_presence(self.position)
        if police_presence > 0.2:
            crime_chance /= (police_presence * 5)
        
        # Decide whether to commit crime
        if random.random() < crime_chance:
            # Find a suitable target
            if random.choice([CrimeType.THEFT, CrimeType.BURGLARY]) == CrimeType.THEFT:
                # Find a citizen to steal from
                potential_victims = [
                    c for c in self.city.citizens 
                    if c != self and c.position.horizontal_distance_to(self.position) < 10
                ]
                if potential_victims:
                    victim = random.choice(potential_victims)
                    self.target_victim = victim
                    self.state = CitizenState.COMMITTING_CRIME
                    self.current_crime = CrimeType.THEFT
            else:
                # Find a building to burglarize
                target = self.city.find_nearest_building(
                    self.position, random.choice([
                        BuildingType.RESIDENTIAL, 
                        BuildingType.COMMERCIAL
                    ])
                )
                if target:
                    entrance = target.get_entrance_position()
                    self._set_target(entrance.x, entrance.y, entrance.z)
                    self.target_building = target
                    self.state = CitizenState.COMMUTING
                    self.next_state = CitizenState.COMMITTING_CRIME
                    self.current_crime = CrimeType.BURGLARY
    
    def _commit_crime(self):
        """Execute the decided crime in 3D space."""
        crime_successful = False
        
        # Check for police presence
        police_presence = self.city.get_police_presence(self.position)
        caught_chance = police_presence * 0.8
        
        if random.random() < caught_chance:
            # Caught by police
            self.state = CitizenState.ARRESTED
            self.arrest_time = 100
            if self.current_crime:
                self.criminal_record.append(self.current_crime)
            return
            
        # Execute crime based on type
        if self.current_crime == CrimeType.THEFT and hasattr(self, 'target_victim'):
            # Steal money from victim
            steal_amount = min(50, self.target_victim.money * 0.3)
            self.target_victim.money -= steal_amount
            self.money += steal_amount
            self.needs[NeedType.INCOME] += 10
            crime_successful = True
            
            # Log theft directly in city crime stats
            self.city.crimes.append((CrimeType.THEFT, self.position, self.city.time_system.day))
            # Update crime heatmap with theft severity
            x, z = int(self.position.x % self.city.width), int(self.position.z % self.city.depth)
            self.city.crime_heatmap[x, z] += 1.0  # Theft has lower severity
            
            delattr(self, 'target_victim')
            
        elif self.current_crime == CrimeType.BURGLARY and hasattr(self, 'target_building'):
            # Burglarize building
            steal_amount = 100 * self.target_building.wealth
            self.money += steal_amount
            self.needs[NeedType.INCOME] += 20
            crime_successful = True
            
            # Log burglary directly in city crime stats
            self.city.crimes.append((CrimeType.BURGLARY, self.position, self.city.time_system.day))
            # Update crime heatmap with burglary severity
            x, z = int(self.position.x % self.city.width), int(self.position.z % self.city.depth)
            self.city.crime_heatmap[x, z] += 2.0  # Burglary has medium severity
            
            delattr(self, 'target_building')
        
        # Reset state and handle aftermath
        self.state = CitizenState.IDLE
        if crime_successful:
            self.crimes_committed += 1
            if random.random() < 0.7 and self.home:
                entrance = self.home.get_entrance_position()
                self._set_target(entrance.x, entrance.y, entrance.z)
                self.state = CitizenState.COMMUTING
    
    def assign_home(self, building, floor=None):
        """Assign a residential building as this citizen's home."""
        if self.home:
            self.home.remove_occupant(self)
        
        self.home = building
        building.add_occupant(self)
        
        # Assign floor if specified, otherwise random
        if floor is None:
            self.home_floor = random.randint(0, building.floors - 1)
        else:
            self.home_floor = min(floor, building.floors - 1)
    
    def assign_job(self, building, floor=None):
        """Assign a workplace to this citizen."""
        if self.workplace:
            self.workplace.remove_employee(self)
        
        self.workplace = building
        building.add_employee(self)
        
        # Assign floor if specified, otherwise random
        if floor is None:
            self.workplace_floor = random.randint(0, building.floors - 1)
        else:
            self.workplace_floor = min(floor, building.floors - 1)
    
    def __str__(self) -> str:
        """Return a string representation of the citizen."""
        return (f"{self.name} at {self.position}, "
                f"State: {self.state.name}, "
                f"Money: ${self.money:.2f}")


# ===== POLICE OFFICER CLASS =====

class PoliceOfficer(Citizen):
    """A police officer citizen who enforces laws and responds to crimes in 3D space."""
    
    def __init__(self, city, x: float, y: float, z: float):
        super().__init__(city, x, y, z)
        self.name = f"Officer-{id(self) % 1000}"
        self.personality = CitizenPersonality.LAWFUL  # Always lawful
        self.patrol_target = None
        self.pursuing_criminal = None
        self.perception_range = 8.0  # Can see crimes within this range
        self.max_speed = 2.2  # Faster than regular citizens
        self.jump_force = 6.0  # Stronger jump
        
        # Override needs
        for need_type in self.needs:
            self.needs[need_type] = 70  # Start with good needs
    
    def update(self, time_system):
        """Update police officer behavior in 3D environment."""
        # Basic need updates
        self._update_needs(time_system)
        
        # Check for nearby crimes or criminals first
        if not self.pursuing_criminal:
            self.pursuing_criminal = self._find_nearby_criminal()
        
        if self.pursuing_criminal:
            # Pursue the criminal
            criminal_pos = self.pursuing_criminal.position
            
            # If there's significant height difference
            if abs(criminal_pos.y - self.position.y) > 1.0:
                # Try to jump if we're below the criminal and on the ground
                if criminal_pos.y > self.position.y and self.grounded:
                    self.jump()
                
                # Set target at criminal's position
                self._set_target(criminal_pos.x, criminal_pos.y, criminal_pos.z)
                
                # Use climbing for vertical movement
                if self.position.horizontal_distance_to(criminal_pos) < 1.0:
                    self.state = CitizenState.CLIMBING
                else:
                    self.state = CitizenState.COMMUTING
            else:
                # Horizontal pursuit
                self._set_target(criminal_pos.x, criminal_pos.y, criminal_pos.z)
                self.state = CitizenState.COMMUTING
            
            # Check if close enough to arrest
            if self.position.distance_to(criminal_pos) < 1.0:
                self._arrest_criminal(self.pursuing_criminal)
                self.pursuing_criminal = None
                self.state = CitizenState.IDLE
        elif self.state == CitizenState.COMMUTING:
            # Continue moving to patrol area
            self._move_to_target()
        elif self.state == CitizenState.CLIMBING:
            # Continue climbing
            self._climb_to_target()
        elif time_system.is_work_hours():
            # On duty - patrol or go to police station
            if random.random() < 0.1 or not self.patrol_target:
                # Find new patrol target
                if random.random() < 0.7:
                    # Patrol a random area on ground level
                    target_x = random.uniform(0, self.city.width - 1)
                    target_z = random.uniform(0, self.city.depth - 1)
                    self._set_target(target_x, 0, target_z)
                else:
                    # Go to high crime area if any
                    hotspot = self.city.find_crime_hotspot()
                    if hotspot:
                        self._set_target(hotspot.x, 0, hotspot.z)
                    else:
                        # Default to random patrol
                        target_x = random.uniform(0, self.city.width - 1)
                        target_z = random.uniform(0, self.city.depth - 1)
                        self._set_target(target_x, 0, target_z)
                        
                self.state = CitizenState.COMMUTING
        else:
            # Off duty behavior - similar to regular citizens
            super()._decide_next_action(time_system)
    
    def _find_nearby_criminal(self):
        """Search for criminals committing crimes nearby in 3D space."""
        for citizen in self.city.citizens:
            if (citizen.state == CitizenState.COMMITTING_CRIME and 
                self.position.horizontal_distance_to(citizen.position) < self.perception_range):
                # Vertical distance check - can't see through floors
                # Assuming each floor is approximately 3 meters
                floor_difference = abs(self.position.y - citizen.position.y) / 3.0
                
                # Can see crime if on same floor or adjacent floor
                if floor_difference <= 1.0:
                    return citizen
        return None
    
    def _arrest_criminal(self, criminal):
        """Arrest a criminal citizen."""
        criminal.state = CitizenState.ARRESTED
        criminal.arrest_time = 100 + (50 * criminal.crimes_committed)  # Longer sentences for repeat offenders
        
        # Add crime to record if known
        if hasattr(criminal, 'current_crime') and criminal.current_crime is not None:
            criminal.criminal_record.append(criminal.current_crime)
        
        # Reset criminal's state
        if hasattr(criminal, 'target_victim'):
            delattr(criminal, 'target_victim')
        if hasattr(criminal, 'target_building'):
            delattr(criminal, 'target_building')
            
        # Log the arrest
        self.city.arrests += 1


# ===== CITY CLASS =====

class City:
    """The main city simulation class containing buildings, citizens, and statistics in 3D space."""
    
    def __init__(self, width: int = 100, depth: int = 100, max_height: int = 50):
        self.width = width  # X dimension
        self.depth = depth  # Z dimension
        self.max_height = max_height  # Y dimension (maximum building height)
        
        # Systems
        self.time_system = TimeSystem()
        self.physics = PhysicsSimulation()
        
        # Map and entities
        self.buildings = []
        self.citizens = []
        self.police_officers = []
        self.police_stations = []
        
        # Statistics
        self.crimes = []  # List of (crime_type, position, time) tuples
        self.crime_heatmap = np.zeros((width, depth))  # 2D, focused on ground level
        self.arrests = 0
        self.economy = {
            'total_money': 0,
            'avg_wage': 0,
            'unemployment_rate': 0.02  # Target 2% unemployment
        }
        
        # Generate city
        self._generate_city()
    
    def _generate_city(self):
        """Generate the initial 3D city layout with buildings."""
        # Place buildings
        self._place_buildings()
        
        # Create initial citizens
        num_citizens = int((self.width * self.depth) * 0.02)  # Lower density for better performance
        self._create_citizens(num_citizens)
        
        # Create police force
        num_police = int(num_citizens * 0.03)  # 3% of population are police
        self._create_police_force(num_police)
        
        # Assign homes and jobs
        self._assign_homes_and_jobs()
    
    def _place_buildings(self):
        """Place buildings throughout the 3D city grid with increased spacing for a sparser layout."""
        # Distribution of building types
        building_dist = {
            BuildingType.RESIDENTIAL: 0.5,
            BuildingType.COMMERCIAL: 0.2,
            BuildingType.INDUSTRIAL: 0.1,
            BuildingType.GOVERNMENT: 0.05,
            BuildingType.POLICE_STATION: 0.02,
            BuildingType.EMPTY: 0.13
        }
        
        # Determine building density - higher in center, lower in outskirts
        building_density = np.zeros((self.width, self.depth))
        
        # Create a gradient from center (high density) to edges (low density)
        center_x, center_z = self.width / 2, self.depth / 2
        for x in range(self.width):
            for z in range(self.depth):
                # Distance from center (normalized to 0-1)
                dist_from_center = np.sqrt((x - center_x)**2 + (z - center_z)**2) / np.sqrt(center_x**2 + center_z**2)
                # Higher values near center, lower at edges
                building_density[x, z] = 1.0 - (dist_from_center * 0.7)
        
        # Increased building spacing for a sparser city
        building_spacing = 20  # Distance between building centers (increased from 10)
        
        for x in range(building_spacing // 2, self.width, building_spacing):
            for z in range(building_spacing // 2, self.depth, building_spacing):
                # Skip more locations based on density + additional sparsity factor
                sparsity_factor = 0.4  # Higher value = fewer buildings (more sparse)
                if random.random() > building_density[x, z] * (1.0 - sparsity_factor):
                    continue
                # Determine district type based on position in city
                if np.sqrt((x - center_x)**2 + (z - center_z)**2) < min(self.width, self.depth) / 5:
                    # Downtown - mixed commercial and government
                    weights = {
                        BuildingType.RESIDENTIAL: 0.2,
                        BuildingType.COMMERCIAL: 0.5,
                        BuildingType.INDUSTRIAL: 0.05,
                        BuildingType.GOVERNMENT: 0.15,
                        BuildingType.POLICE_STATION: 0.05,
                        BuildingType.EMPTY: 0.05
                    }
                elif np.sqrt((x - center_x)**2 + (z - center_z)**2) > min(self.width, self.depth) / 2:
                    # Suburbs - mostly residential
                    weights = {
                        BuildingType.RESIDENTIAL: 0.8,
                        BuildingType.COMMERCIAL: 0.1,
                        BuildingType.INDUSTRIAL: 0.0,
                        BuildingType.GOVERNMENT: 0.0,
                        BuildingType.POLICE_STATION: 0.01,
                        BuildingType.EMPTY: 0.09
                    }
                elif 0.3 < x / self.width < 0.7 and 0.3 < z / self.depth < 0.7:
                    # Industrial zone
                    weights = {
                        BuildingType.RESIDENTIAL: 0.1,
                        BuildingType.COMMERCIAL: 0.2,
                        BuildingType.INDUSTRIAL: 0.6,
                        BuildingType.GOVERNMENT: 0.0,
                        BuildingType.POLICE_STATION: 0.01,
                        BuildingType.EMPTY: 0.09
                    }
                else:
                    # Mixed areas
                    weights = building_dist
                
                # Randomly select building type based on weights
                building_types = list(weights.keys())
                building_weights = [weights[t] for t in building_types]
                building_type = np.random.choice(building_types, p=building_weights)
                
                if building_type != BuildingType.EMPTY:
                    # Determine building size based on type
                    if building_type == BuildingType.RESIDENTIAL:
                        width = random.uniform(6, 12)
                        height = random.uniform(9, 30)  # 3-10 floors
                        depth = random.uniform(6, 12)
                    elif building_type == BuildingType.COMMERCIAL:
                        width = random.uniform(10, 20)
                        height = random.uniform(6, 45)  # 2-15 floors
                        depth = random.uniform(10, 20)
                    elif building_type == BuildingType.INDUSTRIAL:
                        width = random.uniform(15, 30)
                        height = random.uniform(6, 15)  # 2-5 floors
                        depth = random.uniform(15, 30)
                    elif building_type == BuildingType.GOVERNMENT:
                        width = random.uniform(12, 24)
                        height = random.uniform(15, 36)  # 5-12 floors
                        depth = random.uniform(12, 24)
                    elif building_type == BuildingType.POLICE_STATION:
                        width = random.uniform(10, 15)
                        height = random.uniform(9, 15)  # 3-5 floors
                        depth = random.uniform(10, 15)
                    
                    # Create building
                    building = Building(x, 0, z, width, height, depth, building_type)
                    self.buildings.append(building)
                    self.physics.add_entity(building)
                    
                    if building_type == BuildingType.POLICE_STATION:
                        self.police_stations.append(building)
    
    def _create_citizens(self, num_citizens: int):
        """Create the initial citizen population in 3D space."""
        for _ in range(num_citizens):
            # Random position within city at ground level
            x = random.uniform(0, self.width - 1)
            z = random.uniform(0, self.depth - 1)
            
            # Create citizen
            citizen = Citizen(self, x, 0, z)
            self.citizens.append(citizen)
            self.physics.add_entity(citizen)
    
    def _create_police_force(self, num_police: int):
        """Create police officers in 3D space."""
        for _ in range(num_police):
            # Start at police stations if available
            if self.police_stations:
                station = random.choice(self.police_stations)
                entrance = station.get_entrance_position()
                x, y, z = entrance.x, entrance.y, entrance.z
            else:
                x = random.uniform(0, self.width - 1)
                z = random.uniform(0, self.depth - 1)
                y = 0
            
            # Create officer
            officer = PoliceOfficer(self, x, y, z)
            self.citizens.append(officer)
            self.police_officers.append(officer)
            self.physics.add_entity(officer)
            
            # If stations exist, assign workplace
            if self.police_stations:
                station = random.choice(self.police_stations)
                officer.assign_job(station)
    
    def _assign_homes_and_jobs(self):
        """Assign homes and jobs to citizens in 3D buildings."""
        # Get residential buildings
        residential = [b for b in self.buildings if b.type == BuildingType.RESIDENTIAL]
        
        # Get job providing buildings
        workplaces = [
            b for b in self.buildings 
            if b.type in [
                BuildingType.COMMERCIAL, 
                BuildingType.INDUSTRIAL,
                BuildingType.GOVERNMENT,
                BuildingType.POLICE_STATION
            ]
        ]
        
        # Assign homes
        for citizen in self.citizens:
            if residential:
                home = random.choice(residential)
                citizen.assign_home(home)
        
        # Count total available jobs
        total_jobs = sum(b.jobs for b in workplaces)
        
        # Determine how many citizens should be unemployed based on unemployment rate
        num_citizens = len(self.citizens)
        target_employed = int(num_citizens * (1 - self.economy['unemployment_rate']))
        
        # If we don't have enough jobs, adjust the number of jobs in buildings
        if total_jobs < target_employed:
            # Increase jobs in existing workplaces to meet employment target
            jobs_to_add = target_employed - total_jobs
            jobs_per_building = max(1, jobs_to_add // len(workplaces))
            
            for workplace in workplaces:
                workplace.jobs += jobs_per_building
                workplace.job_openings += jobs_per_building
            
        # Assign jobs (ensure we hit target employment rate)
        job_seekers = [c for c in self.citizens if not isinstance(c, PoliceOfficer) or not c.workplace]
        random.shuffle(job_seekers)
        
        # Only assign jobs to the target number of citizens
        for i, citizen in enumerate(job_seekers):
            if i >= target_employed:
                break  # Stop assigning jobs to maintain unemployment rate
                
            if workplaces:
                # Try to find a workplace with openings
                potential_workplaces = [w for w in workplaces if w.job_openings > 0]
                if potential_workplaces:
                    workplace = random.choice(potential_workplaces)
                    citizen.assign_job(workplace)
    
    def update(self):
        """Update the entire city for one time step."""
        # Update time system
        self.time_system.update()
        
        # Update physics simulation
        self.physics.update()
        
        # Update all citizens
        for citizen in self.citizens:
            citizen.update(self.time_system)
        
        # Update economy statistics
        self._update_economy_stats()
        
        # Decay crime heatmap slightly
        self.crime_heatmap *= 0.99
        
        # Occasionally trigger sabotage events
        if random.random() < 0.001:  # Rare sabotage events
            self._trigger_sabotage_event()
            
    def _update_economy_stats(self):
        """Update economic statistics for the city."""
        # Calculate total money in the economy
        self.economy['total_money'] = sum(c.money for c in self.citizens)
        
        # Calculate average wage
        wages = [b.wage for b in self.buildings 
                if b.type in [BuildingType.COMMERCIAL, BuildingType.INDUSTRIAL, 
                             BuildingType.GOVERNMENT, BuildingType.POLICE_STATION]]
        self.economy['avg_wage'] = sum(wages) / max(1, len(wages))
        
        # Calculate unemployment rate (actual)
        working_citizens = sum(1 for c in self.citizens if c.workplace is not None)
        self.economy['unemployment_rate'] = 1.0 - (working_citizens / max(1, len(self.citizens)))
        
    def _trigger_sabotage_event(self):
        """Trigger a major sabotage event targeting city infrastructure."""
        # Find potential targets (preferring industrial and government buildings)
        targets = [b for b in self.buildings 
                  if b.type in [BuildingType.INDUSTRIAL, BuildingType.GOVERNMENT]]
        
        if not targets:
            # Fall back to any building
            targets = self.buildings
            
        if not targets:
            return  # No targets available
            
        # Select a target
        target = random.choice(targets)
        
        # Log the crime directly
        self.crimes.append((CrimeType.SABOTAGE, target.position, self.time_system.day))
        
        # Update crime heatmap
        x, z = int(target.position.x % self.width), int(target.position.z % self.depth)
        
        # Create a hotspot of criminal activity with severity
        severity = 5.0  # Sabotage has high severity
        for dx in range(-3, 4):
            for dz in range(-3, 4):
                nx, nz = (x + dx) % self.width, (z + dz) % self.depth
                distance = np.sqrt(dx**2 + dz**2)
                if distance <= 3:
                    intensity = 1.0 - (distance / 3)
                    self.crime_heatmap[nx, nz] += intensity * severity
        
        # Notify nearby citizens and police
        for citizen in self.citizens:
            if citizen.position.horizontal_distance_to(target.position) < 20:
                citizen.needs[NeedType.SAFETY] = max(0, citizen.needs[NeedType.SAFETY] - 30)
        
        # Dispatch police to investigate
        for officer in self.police_officers:
            if random.random() < 0.3:  # 30% of officers respond
                entrance = target.get_entrance_position()
                officer._set_target(entrance.x, entrance.y, entrance.z)
                officer.state = CitizenState.COMMUTING

    def add_crime(self, crime_type: CrimeType, position: Vector3):
        """Register a crime in the city statistics."""
        # Log the crime
        self.crimes.append((crime_type, position, self.time_system.day))
        
        # Update crime heatmap (2D representation)
        x, z = int(position.x % self.width), int(position.z % self.depth)
        
        # Add to heatmap based on crime severity
        severity = {
            CrimeType.THEFT: 1.0,
            CrimeType.VANDALISM: 1.5,
            CrimeType.ASSAULT: 3.0,
            CrimeType.BURGLARY: 2.0,
            CrimeType.SABOTAGE: 5.0
        }.get(crime_type, 1.0)
        
        self.crime_heatmap[x, z] += severity
        
        # Add to surrounding cells with decay
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                nx, nz = (x + dx) % self.width, (z + dz) % self.depth
                distance = np.sqrt(dx**2 + dz**2)
                if distance <= 2:
                    self.crime_heatmap[nx, nz] += severity * (1 - distance/2)
    
    def get_police_presence(self, position: Vector3) -> float:
        """Calculate police presence (0-1) at a given position."""
        presence = 0.0
        
        for officer in self.police_officers:
            horizontal_distance = position.horizontal_distance_to(officer.position)
            if horizontal_distance < 15:  # Police influence range
                # Police influence decreases with distance
                presence += 1.0 / max(1.0, horizontal_distance)
                
                # Reduce influence if on different floors (height difference)
                height_diff = abs(position.y - officer.position.y)
                if height_diff > 3.0:  # Different floors
                    presence *= 0.5  # Half as effective across floors
                
        # Normalize to 0-1 range (cap at 1.0)
        return min(1.0, presence)
    
    def find_nearest_building(self, position: Vector3, building_type: BuildingType) -> Optional[Building]:
        """Find the nearest building of a specific type in 3D space."""
        candidates = [b for b in self.buildings if b.type == building_type]
        if not candidates:
            return None
            
        # Find nearest
        return min(candidates, key=lambda b: position.horizontal_distance_to(b.position))
    
    def find_random_building_of_type(self, building_types: List[BuildingType]) -> Optional[Building]:
        """Find a random building of one of the specified types."""
        candidates = [b for b in self.buildings if b.type in building_types]
        if not candidates:
            return None
            
        return random.choice(candidates)
    
    def find_crime_hotspot(self) -> Optional[Vector3]:
        """Find a significant crime hotspot to focus police resources."""
        # Find areas with above-average crime rates
        avg_crime = np.mean(self.crime_heatmap)
        threshold = avg_crime * 2  # Focus on areas with double the average crime
        
        # Find cells above threshold
        hotspots = []
        for x in range(self.width):
            for z in range(self.depth):
                if self.crime_heatmap[x, z] > threshold:
                    hotspots.append(Vector3(x, 0, z))  # At ground level
                    
        if not hotspots:
            return None
            
        return random.choice(hotspots)
    
    def get_statistics(self) -> Dict:
        """Get current city statistics for reporting."""
        stats = {
            'time': self.time_system.get_time_string(),
            'population': len(self.citizens),
            'police': len(self.police_officers),
            'buildings': len(self.buildings),
            'crimes_24h': sum(1 for c in self.crimes if c[2] == self.time_system.day),
            'arrests': self.arrests,
            'economy': self.economy,
            'avg_safety': sum(c.needs[NeedType.SAFETY] for c in self.citizens) / max(1, len(self.citizens)),
            'crime_hotspots': sum(1 for x in range(self.width) for z in range(self.depth) 
                               if self.crime_heatmap[x, z] > 2.0)
        }
        return stats


# ===== SIMULATION FUNCTIONS =====

def run_simulation(days=1, steps_per_day=24*6, visualization_interval=12):
    """Run the 3D city simulation for the specified number of days."""
    # Create city with sparser building layout
    city = City(width=100, depth=100, max_height=60)
    
    # Run simulation
    total_steps = days * steps_per_day
    
    for step in range(total_steps):
        # Update city
        city.update()
        
        # Visualize at specific intervals
        if step % visualization_interval == 0:
            clear_output(wait=True)
            print(f"Day {city.time_system.day}, Hour {city.time_system.hour}")
            
            # Create 3D visualization
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Draw a sample of buildings for better performance
            building_sample = random.sample(city.buildings, min(50, len(city.buildings)))
            for building in building_sample:
                # Draw a simplified cuboid for each building
                x, y, z = building.position.x, building.position.y, building.position.z
                w, h, d = building.width, building.height, building.depth
                
                # Determine color based on building type
                if building.type == BuildingType.RESIDENTIAL:
                    color = 'blue'
                elif building.type == BuildingType.COMMERCIAL:
                    color = 'gold'
                elif building.type == BuildingType.INDUSTRIAL:
                    color = 'brown'
                elif building.type == BuildingType.GOVERNMENT:
                    color = 'purple'
                elif building.type == BuildingType.POLICE_STATION:
                    color = 'red'
                else:
                    color = 'gray'
                
                # Create simple cuboid (just the base and top edges)
                ax.plot([x-w/2, x+w/2, x+w/2, x-w/2, x-w/2], 
                        [y, y, y, y, y], 
                        [z-d/2, z-d/2, z+d/2, z+d/2, z-d/2], color=color)
                ax.plot([x-w/2, x+w/2, x+w/2, x-w/2, x-w/2], 
                        [y+h, y+h, y+h, y+h, y+h], 
                        [z-d/2, z-d/2, z+d/2, z+d/2, z-d/2], color=color)
                
                # Connect base to top
                ax.plot([x-w/2, x-w/2], [y, y+h], [z-d/2, z-d/2], color=color)
                ax.plot([x+w/2, x+w/2], [y, y+h], [z-d/2, z-d/2], color=color)
                ax.plot([x+w/2, x+w/2], [y, y+h], [z+d/2, z+d/2], color=color)
                ax.plot([x-w/2, x-w/2], [y, y+h], [z+d/2, z+d/2], color=color)
            
            # Plot citizens and police (sample for performance)
            citizen_sample = random.sample(city.citizens, min(100, len(city.citizens)))
            
            # Regular citizens
            citizen_positions = [(c.position.x, c.position.y, c.position.z) for c in citizen_sample 
                               if not isinstance(c, PoliceOfficer)]
            if citizen_positions:
                xs, ys, zs = zip(*citizen_positions)
                ax.scatter(xs, ys, zs, color='green', s=20, alpha=0.7, label='Citizens')
            
            # Police officers
            police_sample = [c for c in citizen_sample if isinstance(c, PoliceOfficer)]
            police_positions = [(o.position.x, o.position.y, o.position.z) for o in police_sample]
            if police_positions:
                xs, ys, zs = zip(*police_positions)
                ax.scatter(xs, ys, zs, color='red', s=30, alpha=0.9, marker='^', label='Police')
            
            # Set axis limits and labels
            ax.set_xlim(0, city.width)
            ax.set_ylim(0, city.max_height)
            ax.set_zlim(0, city.depth)
            ax.set_xlabel('X')
            ax.set_ylabel('Y (Height)')
            ax.set_zlabel('Z')
            ax.set_title(f"3D City Simulation - {city.time_system.get_time_string()}")
            
            # Add a legend
            ax.legend()
            
            # Print statistics
            stats = city.get_statistics()
            stats_text = (
                f"Population: {stats['population']}\n"
                f"Buildings: {stats['buildings']}\n"
                f"Police: {stats['police']}\n"
                f"Recent Crimes: {stats['crimes_24h']}\n"
                f"Arrests: {stats['arrests']}\n"
                f"Unemployment: {stats['economy']['unemployment_rate']:.1%}\n"
                f"Average Safety: {stats['avg_safety']:.1f}/100"
            )
            print(stats_text)
            
            plt.tight_layout()
            plt.show()
            
            # Slight pause to allow viewing
            time.sleep(0.5)

    return city


def top_down_view(city):
    """Create a 2D top-down view of the city with crime heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw building footprints
    for building in city.buildings:
        x, z = building.position.x, building.position.z
        w, d = building.width, building.depth
        
        # Determine color based on building type
        if building.type == BuildingType.RESIDENTIAL:
            color = 'blue'
        elif building.type == BuildingType.COMMERCIAL:
            color = 'gold'
        elif building.type == BuildingType.INDUSTRIAL:
            color = 'brown'
        elif building.type == BuildingType.GOVERNMENT:
            color = 'purple'
        elif building.type == BuildingType.POLICE_STATION:
            color = 'red'
        else:
            color = 'gray'
            
        # Draw building footprint
        rect = plt.Rectangle((x-w/2, z-d/2), w, d, color=color, alpha=0.6)
        ax.add_patch(rect)
    
    # Show crime heatmap
    # Normalize the heatmap for visualization
    if np.max(city.crime_heatmap) > 0:
        crime_normalized = city.crime_heatmap / np.max(city.crime_heatmap)
        ax.imshow(crime_normalized.T, cmap='Reds', alpha=0.3, extent=[0, city.width, 0, city.depth], origin='lower')
    
    # Plot citizen positions
    citizen_positions = [(c.position.x, c.position.z) for c in city.citizens 
                       if not isinstance(c, PoliceOfficer)]
    if citizen_positions:
        xs, zs = zip(*citizen_positions)
        ax.scatter(xs, zs, color='green', s=10, alpha=0.7, label='Citizens')
    
    # Plot police positions
    police_positions = [(p.position.x, p.position.z) for p in city.police_officers]
    if police_positions:
        xs, zs = zip(*police_positions)
        ax.scatter(xs, zs, color='red', s=20, alpha=0.9, marker='^', label='Police')
    
    # Set limits and labels
    ax.set_xlim(0, city.width)
    ax.set_ylim(0, city.depth)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title(f"City Map - {city.time_system.get_time_string()}")
    ax.legend()
    
    plt.tight_layout()
    return fig


# Quick demo function
def quick_demo():
    """Run a quick demonstration of the 3D city simulation."""
    # Create a smaller city for better performance
    city = City(width=80, depth=80, max_height=50)
    
    # Run for just a few simulation steps
    for _ in range(10):
        city.update()
    
    # Create visualizations
    plt.figure(figsize=(18, 10))
    
    # 3D view
    plt.subplot(1, 2, 1, projection='3d')
    ax = plt.gca()
    
    # Sample buildings for better performance
    building_sample = random.sample(city.buildings, min(30, len(city.buildings)))
    for building in building_sample:
        x, y, z = building.position.x, building.position.y, building.position.z
        w, h, d = building.width, building.height, building.depth
        
        # Color based on building type
        if building.type == BuildingType.RESIDENTIAL:
            color = 'blue'
        elif building.type == BuildingType.COMMERCIAL:
            color = 'gold'
        elif building.type == BuildingType.INDUSTRIAL:
            color = 'brown'
        elif building.type == BuildingType.GOVERNMENT:
            color = 'purple'
        elif building.type == BuildingType.POLICE_STATION:
            color = 'red'
        else:
            color = 'gray'
        
        # Simplified cuboid
        ax.plot([x-w/2, x+w/2, x+w/2, x-w/2, x-w/2], 
                [y, y, y, y, y], 
                [z-d/2, z-d/2, z+d/2, z+d/2, z-d/2], color=color)
        ax.plot([x-w/2, x+w/2, x+w/2, x-w/2, x-w/2], 
                [y+h, y+h, y+h, y+h, y+h], 
                [z-d/2, z-d/2, z+d/2, z+d/2, z-d/2], color=color)
        
        # Corner lines
        ax.plot([x-w/2, x-w/2], [y, y+h], [z-d/2, z-d/2], color=color)
        ax.plot([x+w/2, x+w/2], [y, y+h], [z-d/2, z-d/2], color=color)
        ax.plot([x+w/2, x+w/2], [y, y+h], [z+d/2, z+d/2], color=color)
        ax.plot([x-w/2, x-w/2], [y, y+h], [z+d/2, z+d/2], color=color)
    
    # Sample citizens for better performance
    citizen_sample = random.sample(city.citizens, min(50, len(city.citizens)))
    
    # Plot regular citizens
    citizen_positions = [(c.position.x, c.position.y, c.position.z) for c in citizen_sample 
                       if not isinstance(c, PoliceOfficer)]
    if citizen_positions:
        xs, ys, zs = zip(*citizen_positions)
        ax.scatter(xs, ys, zs, color='green', s=20, alpha=0.7, label='Citizens')
    
    # Plot police officers
    police_sample = [c for c in citizen_sample if isinstance(c, PoliceOfficer)]
    police_positions = [(o.position.x, o.position.y, o.position.z) for o in police_sample]
    if police_positions:
        xs, ys, zs = zip(*police_positions)
        ax.scatter(xs, ys, zs, color='red', s=30, alpha=0.9, marker='^', label='Police')
    
    # Set axis properties
    ax.set_xlim(0, city.width)
    ax.set_ylim(0, city.max_height)
    ax.set_zlim(0, city.depth)
    ax.set_xlabel('X')
    ax.set_ylabel('Y (Height)')
    ax.set_zlabel('Z')
    ax.set_title("3D City View")
    ax.legend()
    
    # Top-down view in the second subplot
    plt.subplot(1, 2, 2)
    ax2 = plt.gca()
    
    # Draw building footprints
    for building in building_sample:
        x, z = building.position.x, building.position.z
        w, d = building.width, building.depth
        
        # Color based on building type
        if building.type == BuildingType.RESIDENTIAL:
            color = 'blue'
        elif building.type == BuildingType.COMMERCIAL:
            color = 'gold'
        elif building.type == BuildingType.INDUSTRIAL:
            color = 'brown'
        elif building.type == BuildingType.GOVERNMENT:
            color = 'purple'
        elif building.type == BuildingType.POLICE_STATION:
            color = 'red'
        else:
            color = 'gray'
            
        # Draw building footprint
        rect = plt.Rectangle((x-w/2, z-d/2), w, d, color=color, alpha=0.6)
        ax2.add_patch(rect)
    
    # Plot citizen positions
    if citizen_positions:
        xs, _, zs = zip(*citizen_positions)
        ax2.scatter(xs, zs, color='green', s=10, alpha=0.7, label='Citizens')
    
    # Plot police positions
    if police_positions:
        xs, _, zs = zip(*police_positions)
        ax2.scatter(xs, zs, color='red', s=20, alpha=0.9, marker='^', label='Police')
    
    # Set properties
    ax2.set_xlim(0, city.width)
    ax2.set_ylim(0, city.depth)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title("Top-Down City Map")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    stats = city.get_statistics()
    print(f"City Statistics:")
    print(f"Population: {stats['population']}")
    print(f"Buildings: {stats['buildings']}")
    print(f"Police: {stats['police']}")
    print(f"Unemployment: {stats['economy']['unemployment_rate']:.1%}")
    
    return city

# Uncomment one of these to run the simulation:
city = run_simulation(days=365, steps_per_day=24, visualization_interval=6)
# city = quick_demo()
