import random
import math
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np

# Vector3 class for position and movement
class Vector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
    
    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

# Time of day enum
class TimeOfDay(Enum):
    MORNING = 0
    AFTERNOON = 1
    EVENING = 2
    NIGHT = 3

# Building types
class BuildingType(Enum):
    RESIDENTIAL = 0
    COMMERCIAL = 1
    INDUSTRIAL = 2
    GOVERNMENT = 3
    PARK = 4

class Building:
    """Represents a building in the city"""
    
    def __init__(self, x, y, building_type, size=1.0):
        self.position = Vector3(x, y, 0)
        self.type = building_type
        self.size = size  # Size/scale of building
        self.occupants = []
        self.max_occupants = int(size * 10)  # Bigger buildings can hold more people
        self.wealth = 50.0 * size  # Starting wealth based on size
        
        # Economic properties based on building type
        if building_type == BuildingType.COMMERCIAL:
            self.job_slots = int(size * 5)
            self.goods_produced = 0
            self.wages = 20.0
        elif building_type == BuildingType.INDUSTRIAL:
            self.job_slots = int(size * 8)
            self.goods_produced = 0
            self.wages = 15.0
        elif building_type == BuildingType.GOVERNMENT:
            self.job_slots = int(size * 3)
            self.goods_produced = 0
            self.wages = 25.0
        else:
            self.job_slots = 0
            self.goods_produced = 0
            self.wages = 0.0
    
    def has_space(self):
        """Check if building has space for more occupants"""
        return len(self.occupants) < self.max_occupants
    
    def add_occupant(self, citizen):
        """Add occupant to building"""
        if self.has_space():
            self.occupants.append(citizen)
            return True
        return False
    
    def remove_occupant(self, citizen):
        """Remove occupant from building"""
        if citizen in self.occupants:
            self.occupants.remove(citizen)
            return True
        return False
    
    def update(self, city):
        """Update building state"""
        # Commercial and industrial buildings produce goods/services
        if self.type in [BuildingType.COMMERCIAL, BuildingType.INDUSTRIAL]:
            workers = [c for c in self.occupants if c.current_activity == "working"]
            # Production based on number of workers
            production = len(workers) * 2.0
            self.goods_produced += production
            
            # Pay workers
            for worker in workers:
                payment = self.wages
                if self.wealth >= payment:
                    self.wealth -= payment
                    worker.money += payment
                else:
                    # Pay what we can if building doesn't have enough money
                    worker.money += self.wealth
                    self.wealth = 0
            
            # Sell goods/services
            if self.goods_produced > 0:
                sales = min(self.goods_produced, len(self.occupants) * 0.5)
                self.goods_produced -= sales
                self.wealth += sales * 5.0  # Revenue from sales

class Citizen:
    """An individual citizen agent in the city simulation"""
    
    def __init__(self, x=0.0, y=0.0):
        self.position = Vector3(x, y, 0)
        self.home = None
        self.workplace = None
        self.money = random.uniform(100, 500)
        self.education = random.uniform(0, 10)
        self.happiness = random.uniform(50, 100)
        self.health = random.uniform(70, 100)
        self.age = random.randint(18, 65)
        self.current_activity = "idle"
        self.current_building = None
        self.move_target = None
        self.speed = random.uniform(0.8, 1.5)
        self.needs = {
            "hunger": random.uniform(0, 50),
            "rest": random.uniform(0, 50),
            "entertainment": random.uniform(0, 50)
        }
    
    def update(self, city):
        """Update citizen state and make decisions"""
        self._update_needs()
        
        # Make decisions based on time of day
        if city.time_of_day == TimeOfDay.MORNING:
            self._morning_routine(city)
        elif city.time_of_day == TimeOfDay.AFTERNOON:
            self._afternoon_routine(city)
        elif city.time_of_day == TimeOfDay.EVENING:
            self._evening_routine(city)
        else:  # NIGHT
            self._night_routine(city)
        
        # Move toward target if one exists
        self._move_toward_target()
        
        # Update happiness based on needs and economic status
        self._update_happiness()
    
    def _update_needs(self):
        """Update citizen needs"""
        self.needs["hunger"] += random.uniform(0.5, 1.5)
        self.needs["rest"] += random.uniform(0.3, 1.0)
        self.needs["entertainment"] += random.uniform(0.2, 0.8)
        
        # Cap needs at 100
        for need in self.needs:
            self.needs[need] = min(self.needs[need], 100)
    
    def _morning_routine(self, city):
        """Morning activities (go to work)"""
        if self.current_activity == "sleeping":
            self.current_activity = "idle"
            self.needs["rest"] = max(0, self.needs["rest"] - 50)
        
        # Go to work if citizen has a job
        if self.workplace and self.current_building != self.workplace:
            self.current_activity = "traveling"
            self.move_target = self.workplace.position
            self._exit_current_building()
        elif self.workplace and self.current_building == self.workplace:
            self.current_activity = "working"
        # Look for work if unemployed
        elif not self.workplace and random.random() < 0.3:
            self._find_job(city)
    
    def _afternoon_routine(self, city):
        """Afternoon activities (work and shopping)"""
        # If at work, keep working
        if self.current_building == self.workplace:
            self.current_activity = "working"
        # If hungry, go shopping
        elif self.needs["hunger"] > 70 and self.money > 20:
            self._find_and_go_to_building(city, BuildingType.COMMERCIAL)
            if self.current_building and self.current_building.type == BuildingType.COMMERCIAL:
                self.current_activity = "shopping"
                # Buy food
                cost = random.uniform(10, 30)
                if self.money >= cost:
                    self.money -= cost
                    self.needs["hunger"] = max(0, self.needs["hunger"] - 60)
        # Go to work if not there yet
        elif self.workplace and self.current_building != self.workplace:
            self.current_activity = "traveling"
            self.move_target = self.workplace.position
    
    def _evening_routine(self, city):
        """Evening activities (recreation and going home)"""
        # If need entertainment, go to park
        if self.needs["entertainment"] > 60 and random.random() < 0.4:
            self._find_and_go_to_building(city, BuildingType.PARK)
            if self.current_building and self.current_building.type == BuildingType.PARK:
                self.current_activity = "recreation"
                self.needs["entertainment"] = max(0, self.needs["entertainment"] - 40)
        # Head home
        elif self.home and self.current_building != self.home:
            self.current_activity = "traveling"
            self.move_target = self.home.position
            self._exit_current_building()
        elif self.home and self.current_building == self.home:
            self.current_activity = "resting"
    
    def _night_routine(self, city):
        """Night activities (sleep at home)"""
        # Try to go home if not already there
        if self.home and self.current_building != self.home:
            self.current_activity = "traveling"
            self.move_target = self.home.position
            self._exit_current_building()
        elif self.home and self.current_building == self.home:
            self.current_activity = "sleeping"
            # Reduce rest need when sleeping
            self.needs["rest"] = max(0, self.needs["rest"] - 10)
    
    def _move_toward_target(self):
        """Move citizen toward current movement target"""
        if not self.move_target or self.current_building:
            return
        
        # Calculate direction to target
        dx = self.move_target.x - self.position.x
        dy = self.move_target.y - self.position.y
        distance = (dx**2 + dy**2)**0.5
        
        # If close to target, arrive at location
        if distance < 0.5:
            self.position.x = self.move_target.x
            self.position.y = self.move_target.y
            self.move_target = None
            return
        
        # Move toward target
        if distance > 0:
            self.position.x += (dx / distance) * self.speed
            self.position.y += (dy / distance) * self.speed
    
    def _update_happiness(self):
        """Update happiness based on needs and conditions"""
        # Calculate overall need satisfaction (0-100 scale)
        need_satisfaction = 100 - sum(self.needs.values()) / len(self.needs)
        
        # Calculate economic satisfaction
        economic_satisfaction = min(100, self.money / 10)
        
        # Update happiness
        target_happiness = (need_satisfaction * 0.6) + (economic_satisfaction * 0.4)
        
        # Happiness changes gradually
        happiness_change = (target_happiness - self.happiness) * 0.1
        self.happiness += happiness_change
        
        # Ensure happiness stays in range
        self.happiness = max(0, min(100, self.happiness))
    
    def _find_job(self, city):
        """Look for employment"""
        # Find buildings with job openings
        potential_employers = []
        for building in city.buildings:
            if building.type in [BuildingType.COMMERCIAL, BuildingType.INDUSTRIAL, BuildingType.GOVERNMENT]:
                if hasattr(building, 'job_slots'):
                    # Count current workers
                    current_workers = sum(1 for citizen in city.citizens if citizen.workplace == building)
                    if current_workers < building.job_slots:
                        potential_employers.append(building)
        
        # If there are potential employers, pick one based on wages and distance
        if potential_employers:
            # Simple scoring system for job selection
            def job_score(building):
                # Distance penalty
                dx = building.position.x - self.position.x
                dy = building.position.y - self.position.y
                distance = (dx**2 + dy**2)**0.5
                distance_score = max(0, 100 - distance)
                
                # Wage score
                wage_score = building.wages * 4
                
                return distance_score + wage_score
            
            # Sort by score and get the best job
            potential_employers.sort(key=job_score, reverse=True)
            self.workplace = potential_employers[0]
            return True
        
        return False
    
    def _find_and_go_to_building(self, city, building_type):
        """Find and go to a building of specified type"""
        # If already in a building of this type, stay there
        if self.current_building and self.current_building.type == building_type:
            return True
        
        # Find buildings of specified type
        target_buildings = [b for b in city.buildings if b.type == building_type and b.has_space()]
        
        if target_buildings:
            # Sort by distance
            target_buildings.sort(key=lambda b: 
                ((b.position.x - self.position.x)**2 + 
                 (b.position.y - self.position.y)**2))
            
            # Go to closest building
            self._exit_current_building()
            self.move_target = target_buildings[0].position
            self.current_activity = "traveling"
            return True
        
        return False
    
    def _exit_current_building(self):
        """Leave current building if in one"""
        if self.current_building:
            self.current_building.remove_occupant(self)
            self.current_building = None
    
    def enter_building(self, building):
        """Enter a building"""
        if building.add_occupant(self):
            self._exit_current_building()  # Leave current building if in one
            self.current_building = building
            self.position.x = building.position.x
            self.position.y = building.position.y
            self.move_target = None
            return True
        return False

class City:
    """City simulation with buildings and citizens"""
    
    def __init__(self, width=100.0, height=100.0):
        self.width = width
        self.height = height
        self.citizens = []
        self.buildings = []
        self.day = 0
        self.time_of_day = TimeOfDay.MORNING
        self.time_step = 0
        self.economy = {
            "total_wealth": 0,
            "average_income": 0,
            "unemployment_rate": 0,
            "happiness_index": 0
        }
    
    def add_building(self, x, y, building_type, size=1.0):
        """Add a new building to the city"""
        building = Building(x, y, building_type, size)
        self.buildings.append(building)
        return building
    
    def add_citizen(self, x=None, y=None):
        """Add a new citizen to the city"""
        # If no position specified, place randomly
        if x is None:
            x = random.uniform(0, self.width)
        if y is None:
            y = random.uniform(0, self.height)
        
        citizen = Citizen(x, y)
        self.citizens.append(citizen)
        
        # Try to assign a home
        residential_buildings = [b for b in self.buildings if b.type == BuildingType.RESIDENTIAL and b.has_space()]
        if residential_buildings:
            home = random.choice(residential_buildings)
            citizen.home = home
            # Move the citizen in their home initially
            citizen.enter_building(home)
        
        return citizen
    
    def update(self):
        """Update the city for one time step"""
        self.time_step += 1
        
        # Update time of day every 6 time steps (4 periods per day)
        if self.time_step % 6 == 0:
            self.time_of_day = TimeOfDay((self.time_of_day.value + 1) % 4)
            if self.time_of_day == TimeOfDay.MORNING:
                self.day += 1
        
        # Update all buildings
        for building in self.buildings:
            building.update(self)
        
        # Update all citizens
        for citizen in self.citizens:
            citizen.update(self)
            
            # Check if citizen has arrived at a building
            if not citizen.move_target and not citizen.current_building:
                # Find the closest building
                closest_building = None
                closest_distance = float('inf')
                
                for building in self.buildings:
                    dx = building.position.x - citizen.position.x
                    dy = building.position.y - citizen.position.y
                    distance = (dx**2 + dy**2)**0.5
                    
                    if distance < closest_distance and distance < 1.0:  # Within 1.0 unit
                        closest_building = building
                        closest_distance = distance
                
                # Enter the building if one is found
                if closest_building:
                    citizen.enter_building(closest_building)
        
        # Update economic indicators
        self._update_economy()
    
    def _update_economy(self):
        """Update economic indicators"""
        # Total wealth (citizens + buildings)
        citizen_wealth = sum(c.money for c in self.citizens)
        building_wealth = sum(b.wealth for b in self.buildings)
        self.economy["total_wealth"] = citizen_wealth + building_wealth
        
        # Average income (based on employed citizens)
        employed_citizens = [c for c in self.citizens if c.workplace]
        if employed_citizens:
            income_sum = sum(c.money for c in employed_citizens) / len(employed_citizens)
            # Smooth the average income
            if self.economy["average_income"] == 0:
                self.economy["average_income"] = income_sum
            else:
                self.economy["average_income"] = 0.95 * self.economy["average_income"] + 0.05 * income_sum
        
        # Unemployment rate
        if self.citizens:
            unemployed = sum(1 for c in self.citizens if not c.workplace)
            self.economy["unemployment_rate"] = (unemployed / len(self.citizens)) * 100
        
        # Happiness index
        if self.citizens:
            avg_happiness = sum(c.happiness for c in self.citizens) / len(self.citizens)
            self.economy["happiness_index"] = avg_happiness
    
    def simulate(self, steps=10, visualize=False):
        """Run simulation for specified number of steps"""
        history = {
            "days": [],
            "wealth": [],
            "income": [],
            "unemployment": [],
            "happiness": []
        }
        
        for _ in range(steps):
            self.update()
            
            # Record data every morning (once per day)
            if self.time_of_day == TimeOfDay.MORNING:
                history["days"].append(self.day)
                history["wealth"].append(self.economy["total_wealth"])
                history["income"].append(self.economy["average_income"])
                history["unemployment"].append(self.economy["unemployment_rate"])
                history["happiness"].append(self.economy["happiness_index"])
                
                if visualize:
                    self.visualize()
        
        return history
    
    def visualize(self):
        """Visualize current state of the city"""
        plt.figure(figsize=(12, 10))
        
        # Plot buildings
        for building in self.buildings:
            color = {
                BuildingType.RESIDENTIAL: 'blue',
                BuildingType.COMMERCIAL: 'red',
                BuildingType.INDUSTRIAL: 'gray',
                BuildingType.GOVERNMENT: 'purple',
                BuildingType.PARK: 'green'
            }.get(building.type, 'black')
            
            size = building.size * 100
            plt.scatter(building.position.x, building.position.y, 
                      c=color, s=size, alpha=0.6, marker='s')
        
        # Plot citizens
        citizen_x = [c.position.x for c in self.citizens if not c.current_building]
        citizen_y = [c.position.y for c in self.citizens if not c.current_building]
        
        if citizen_x:  # Only plot if there are citizens to show
            plt.scatter(citizen_x, citizen_y, c='orange', s=20, alpha=0.7)
        
        # City boundaries
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        
        # Add info
        plt.title(f"City Simulation - Day {self.day}, {self.time_of_day.name}")
        plt.figtext(0.02, 0.02, 
                  f"Population: {len(self.citizens)}\n"
                  f"Unemployment: {self.economy['unemployment_rate']:.1f}%\n"
                  f"Happiness: {self.economy['happiness_index']:.1f}/100\n"
                  f"Avg Income: ${self.economy['average_income']:.2f}")
        
        plt.show()

# Example usage
def create_sample_city():
    """Create a sample city with buildings and citizens"""
    city = City(width=50, height=50)
    
    # Add buildings
    # Residential district
    for _ in range(10):
        city.add_building(
            random.uniform(5, 15),
            random.uniform(5, 45),
            BuildingType.RESIDENTIAL,
            random.uniform(0.8, 1.5)
        )
    
    # Commercial district
    for _ in range(5):
        city.add_building(
            random.uniform(20, 30),
            random.uniform(10, 40),
            BuildingType.COMMERCIAL,
            random.uniform(1.0, 2.0)
        )
    
    # Industrial district
    for _ in range(3):
        city.add_building(
            random.uniform(35, 45),
            random.uniform(10, 40),
            BuildingType.INDUSTRIAL,
            random.uniform(1.5, 2.5)
        )
    
    # Government buildings
    city.add_building(25, 25, BuildingType.GOVERNMENT, 2.0)
    
    # Parks
    city.add_building(15, 30, BuildingType.PARK, 1.5)
    city.add_building(35, 15, BuildingType.PARK, 1.0)
    
    # Add citizens
    for _ in range(50):
        city.add_citizen()
    
    return city

# Run a simulation
if __name__ == "__main__":
    city = create_sample_city()
    print(f"Created city with {len(city.citizens)} citizens and {len(city.buildings)} buildings")
    
    # Run simulation for 1 day (24 time steps)
    history = city.simulate(steps=24, visualize=True)
    
    # Plot economic trends
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(history["days"], history["wealth"])
    plt.title("Total City Wealth")
    plt.xlabel("Day")
    plt.ylabel("Wealth")
    
    plt.subplot(2, 2, 2)
    plt.plot(history["days"], history["income"])
    plt.title("Average Income")
    plt.xlabel("Day")
    plt.ylabel("Income")
    
    plt.subplot(2, 2, 3)
    plt.plot(history["days"], history["unemployment"])
    plt.title("Unemployment Rate")
    plt.xlabel("Day")
    plt.ylabel("Percentage")
    
    plt.subplot(2, 2, 4)
    plt.plot(history["days"], history["happiness"])
    plt.title("Happiness Index")
    plt.xlabel("Day")
    plt.ylabel("Happiness (0-100)")
    
    plt.tight_layout()
    plt.show()