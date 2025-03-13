import numpy as np

class Vector3:
    """A simple 3D vector class for position, velocity, and acceleration."""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


class Entity:
    """An entity in the physics simulation with physical properties."""
    def __init__(self, x=0.0, y=10.0, z=0.0, mass=1.0):
        self.position = Vector3(x, y, z)
        self.velocity = Vector3(0, 0, 0)
        self.acceleration = Vector3(0, 0, 0)
        self.mass = mass
        self.restitution = 0.7  # Bounciness factor (0 = no bounce, 1 = perfect bounce)
        self.grounded = False


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
    
    def add_entity(self, entity):
        """Add an entity to the simulation."""
        # Ensure entity is at or above ground level
        if entity.position.y < self.ground_level:
            entity.position.y = self.ground_level
        
        self.entities.append(entity)
        return entity
    
    def create_entity(self, x=0.0, y=10.0, z=0.0, mass=1.0):
        """Create a new entity with physics properties."""
        entity = Entity(x, y, z, mass)
        return self.add_entity(entity)
    
    def apply_force(self, entity, force_x, force_y, force_z):
        """Apply a force to an entity."""
        # F = ma, so a = F/m
        entity.acceleration.x += force_x / entity.mass
        entity.acceleration.y += force_y / entity.mass
        entity.acceleration.z += force_z / entity.mass
        
        # Apply acceleration to velocity
        entity.velocity.x += entity.acceleration.x * self.time_step
        entity.velocity.y += entity.acceleration.y * self.time_step
        entity.velocity.z += entity.acceleration.z * self.time_step
        
        # Reset acceleration
        entity.acceleration.x = 0
        entity.acceleration.y = 0
        entity.acceleration.z = 0
    
    def is_grounded(self, entity):
        """Check if an entity is on the ground."""
        return abs(entity.position.y - self.ground_level) < 0.01 and entity.velocity.y <= 0
    
    def update(self):
        """Apply gravitational force to all entities and update their positions."""
        for entity in self.entities:
            # Only apply gravity if not grounded or if about to leave the ground (positive velocity)
            if not entity.grounded or entity.velocity.y > 0:
                entity.velocity.y -= self.gravity * self.time_step
            
            # Update position based on velocity
            entity.position.x += entity.velocity.x * self.time_step
            entity.position.y += entity.velocity.y * self.time_step
            entity.position.z += entity.velocity.z * self.time_step
            
            # Check for ground collision
            if entity.position.y <= self.ground_level:
                entity.position.y = self.ground_level
                entity.grounded = True
                
                # Bounce if moving downward
                if entity.velocity.y < 0:
                    entity.velocity.y = -entity.velocity.y * entity.restitution
                    
                    # Apply friction to horizontal movement
                    entity.velocity.x *= 0.95
                    entity.velocity.z *= 0.95
                    
                    # Stop very small bounces
                    if abs(entity.velocity.y) < 0.1:
                        entity.velocity.y = 0
            else:
                entity.grounded = False
            



def run_simulation(steps=100, step_size=10):
    """Run the simulation for a specified number of steps."""
    # Create the simulation
    simulation = PhysicsSimulation()
    
    # Create an entity (starting at x=0, y=10, z=0 with mass=1)
    agent = simulation.create_entity()
    
    print(f"Starting simulation with gravity: {simulation.gravity} m/s²")
    print(f"Step 0: Position {agent.position}, "
          f"Velocity: ({agent.velocity.x:.2f}, {agent.velocity.y:.2f}, {agent.velocity.z:.2f}), "
          f"Grounded: {agent.grounded}")
    
    for i in range(1, steps+1):
        simulation.update()
        
        # Log based on specified step size
        if i % step_size == 0:
            print(f"Step {i}: Position {agent.position}, "
                  f"Velocity: ({agent.velocity.x:.2f}, {agent.velocity.y:.2f}, {agent.velocity.z:.2f}), "
                  f"Grounded: {agent.grounded}")
    
    print("Simulation complete")


if __name__ == "__main__":
    run_simulation()