import gym
from gym import Env, spaces
import numpy as np
import pygame
import random
import time
import math
import folium
from selenium import webdriver
from PIL import Image
import os
import copy
from typing import List

# # # # !pip install gym
# # # !pip install pygame
#pip install folium
# !pip install selenium

import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class Ship:
    def __init__(self, environment, ship_id: int, lat: float = 0.0, lon: float = 0.0, speed=5, screen_width: int = 600, screen_height: int = 400,
                 ship_type: str = 'ship', firing_range: int = 100, ship_health: int = 100,
                 reload_delay: float = 0.5, target_delay: float = 0.2, helicop_count: int = 0, torpedo_count: int = 100,
                 torpedo_fire_speed: float = 2.0, torpedo_damage: int = 1, decoyM_count: int = 0,
                 decoyM_speed: float = 4.0, decoyM_blast_range: float = 2.0):

        self.ship_id = ship_id
        self.lat = lat
        self.lon = lon
        self.x = 0
        self.y = 0
        self.speed = speed
        self.width = screen_width
        self.height = screen_height
        self.ship_type = ship_type
        self.ship_health = ship_health
        self.firing_range = firing_range
        self.reload_delay = reload_delay
        self.target_delay = target_delay

        # details ffor torpedo
        self.torpedo_fire_speed = torpedo_fire_speed
        self.torpedo_count = torpedo_count
        self.torpedo_damage = torpedo_damage
        self.torpedoes = []  # Store active torpedoes
        self.last_fire_time = 0
        self.target_lock_time = 0

        # for Decoy missile
        self.decoyM_count = decoyM_count
        self.decoy_missile = []  # Store active decoy_missile
        self.decoyM_speed = decoyM_speed
        self.decoyM_blast_range = decoyM_blast_range
        self.last_decoy_fire_time = 0
        self.decoy_target_lock_time = 0


        self.helicop_count = helicop_count
        self.env = environment

    def update_pixel_position(self):
        self.x, self.y = self.env.mapGenerator._latlon_to_pixels(self.lat, self.lon)
        self.x = np.clip(self.x, 0, self.width - 1)
        self.y = np.clip(self.y, 0, self.height - 1)



    def set_position(self, x: int, y: int) -> bool:
        """Ensures the ship's position stays within screen boundaries."""
        # Clamp coordinates to stay within screen boundaries
        clamped_x = np.clip(x, 0, self.width - 10)
        clamped_y = np.clip(y, 0, self.height - 10)

        self.x, self.y = clamped_x, clamped_y
        return (x == clamped_x and y == clamped_y)


    def move_ship_to_direction(self, heading):
    # Get degrees-per-pixel from current map
        dpp_lat = self.env.mapGenerator.degrees_per_pixel_lat
        dpp_lon = self.env.mapGenerator.degrees_per_pixel_lon

    # Convert speed to geographic delta using zoom scale
        delta_lon = self.speed * np.cos(np.radians(heading)) * dpp_lon
        delta_lat = -self.speed * np.sin(np.radians(heading)) * dpp_lat  # Negative for north/upward

    # Update geographic position
        self.lon += delta_lon
        self.lat += delta_lat

    # Clamp to valid lat/lon (optional)
        self.lat = np.clip(self.lat, -90, 90)
        self.lon = np.clip(self.lon, -180, 180)

    # Update x/y projection from lat/lon
        self.update_pixel_position()



    def move_ship_to_coordinates(self, target_latlon, threshold_km=0.5, angle_increment=45, max_angle_adjustment=90, min_spacing_km=50):
        """
        Move the ship towards a target (lat, lon) using haversine distance and collision-aware redirection.
        Maintains minimum spacing from other ships.
        """
        current_lat, current_lon = self.lat, self.lon
        target_lat, target_lon = target_latlon

        # 1. Calculate distance to target
        distance_km = haversine_distance(current_lat, current_lon, target_lat, target_lon)
        if distance_km <= threshold_km:
            self.lat, self.lon = target_lat, target_lon
            self.update_pixel_position()
            return True  # Target reached

        # 2. Calculate initial bearing from current to target
        bearing_rad = math.atan2(
            math.radians(target_lon - current_lon),
            math.radians(target_lat - current_lat)
        )

        # 3. Try adjusted headings to avoid collisions and maintain spacing
        for adjustment in range(0, max_angle_adjustment + angle_increment, angle_increment):
            for angle_offset in [-adjustment, adjustment]:
                adjusted_bearing = bearing_rad + math.radians(angle_offset)

                # Convert bearing to geographic movement (delta lat/lon)
                delta_lat = self.speed * math.cos(adjusted_bearing) / 111
                delta_lon = self.speed * math.sin(adjusted_bearing) / (111 * math.cos(math.radians(current_lat)))

                candidate_lat = current_lat + delta_lat
                candidate_lon = current_lon + delta_lon

                # Check spacing with other ships
                too_close = False
                for other_ship in self.env.ships:
                    if other_ship != self and other_ship != self.env.hvu:  # Skip self and HVU
                        spacing = haversine_distance(candidate_lat, candidate_lon, 
                                                   other_ship.lat, other_ship.lon)
                        if spacing < min_spacing_km:
                            too_close = True
                            break

                if too_close:
                    continue  # Try next angle if too close to another ship

                # Convert positions to pixels for LoS-based collision check
                current_px = self.env.mapGenerator._latlon_to_pixels(current_lat, current_lon)
                candidate_px = self.env.mapGenerator._latlon_to_pixels(candidate_lat, candidate_lon)
                target_px = self.env.mapGenerator._latlon_to_pixels(target_lat, target_lon)

                ship_in_way = self.env.check_if_blocking_los(current_px, candidate_px, target_px)

                if not ship_in_way:
                    # ✅ Safe to move
                    self.lat = np.clip(candidate_lat, -85.0, 85.0)
                    self.lon = np.clip(candidate_lon, -180.0, 180.0)
                    self.update_pixel_position()
                    return False  # Not yet at target

        # 4. 🚨 Fallback: Only attacker is allowed to force movement even if blocked
        if getattr(self, 'ship_type', '') == 'attacker_ship':
            delta_lat = self.speed * math.cos(bearing_rad) / 111
            delta_lon = self.speed * math.sin(bearing_rad) / (111 * math.cos(math.radians(current_lat)))

            fallback_lat = current_lat + delta_lat
            fallback_lon = current_lon + delta_lon

            self.lat = np.clip(fallback_lat, -85.0, 85.0)
            self.lon = np.clip(fallback_lon, -180.0, 180.0)
            self.update_pixel_position()
            return False

        # ❌ Blocked and not allowed to force movement
        return False


    def get_position(self) -> np.ndarray:
        """Returns the current position of the ship as a numpy array."""
        return np.array([self.x, self.y])


    def take_damage(self, damage: int):
        """Reduces the ship's health by a given damage value and returns reward and done status."""
        self.ship_health = max(0, self.ship_health - damage)
        reward = 100 if self.ship_health == 0 else 50
        done = self.ship_health == 0
        return reward, done


    def target_in_range(self, target_ship, threshold=0):
        """Check if target ship is within firing range using haversine distance."""
        distance = haversine_distance(self.lat, self.lon, target_ship.lat, target_ship.lon)
        # Convert firing range from pixels to km (approximately)
        range_km = (self.firing_range + threshold) / 10  # Approximate conversion
        return distance <= range_km




    def __repr__(self) -> str:
        info = (
            f"Ship(id={self.ship_id}, type={self.ship_type}, x={self.x}, y={self.y}, "
            f"health={self.ship_health}, speed={self.speed}, firing_range={self.firing_range}, "
            f"torpedo_count={self.torpedo_count}, torpedo_speed={self.torpedo_fire_speed}, torpedo_damage={self.torpedo_damage}, "
            f"active_torpedoes={len(self.torpedoes)}, "
            f"decoy_missiles={self.decoyM_count}, "
            f"helicop_count={self.helicop_count}"
        )
        return info

import math
import numpy as np

class Torpedo:
    def __init__(self, torpedo_id, lat, lon, speed, damage, direction,
                 source, target, mapGenerator):
        """
        Initialize a torpedo in geographic coordinates.
        """
        self.id = torpedo_id
        self.lat = lat
        self.lon = lon
        self.speed = speed  # km per step
        self.damage = damage
        self.direction = direction  # Radians (bearing from lat/lon)
        self.source = source
        self.target = target
        self.mapGenerator = mapGenerator

        self.target_hit = False

        # Initialize pixel position for rendering
        self.update_pixel_position()

    def update_pixel_position(self):
        """
        Convert current lat/lon to pixel x/y using map generator.
        """
        self.x, self.y = self.mapGenerator._latlon_to_pixels(self.lat, self.lon)

    def move(self):
        """
        Move the torpedo forward in geographic space using bearing.
        """
        delta_lat = self.speed * math.cos(self.direction) / 111  # 111 km per degree latitude
        delta_lon = self.speed * math.sin(self.direction) / (111 * math.cos(math.radians(self.lat)))

        self.lat += delta_lat
        self.lon += delta_lon

        # Optional clamping (to stay on map)
        self.lat = np.clip(self.lat, -85.0, 85.0)
        self.lon = np.clip(self.lon, -180.0, 180.0)

        self.update_pixel_position()

    def within_bounds(self, screen_width, screen_height):
        """
        Check if the torpedo is still within the screen bounds.
        Returns True if the torpedo is within bounds, False otherwise.
        """
        return 0 <= self.x <= screen_width and 0 <= self.y <= screen_height


    def hit_target(self, threshold=0.5):
        """
        Check if the torpedo has hit its target.
        Uses haversine distance for geographic collision check.
        Threshold is in kilometers.
        """
        distance = haversine_distance(self.lat, self.lon, self.target.lat, self.target.lon)

        if distance < threshold:
            reward, done = self.target.take_damage(self.damage)
            self.target_hit = True
            return reward, done

        return 0, False

    def check_collision(self, ships, threshold=0.5):
        """
        Check for collisions with ships other than the target.
        Uses haversine distance for geographic collision detection.
        Threshold is in kilometers.
        """
        for ship in ships:
            if ship == self.target:
                continue

            distance = haversine_distance(self.lat, self.lon, ship.lat, ship.lon)
            if distance < threshold:
                return True

        return False

    def __repr__(self):
        return (f"Torpedo(id={self.id}, lat={self.lat:.4f}, lon={self.lon:.4f}, "
                f"speed={self.speed}, damage={self.damage}, target_hit={self.target_hit})")

class DefenseSystem():
    def __init__(self, env):
        self.env = env
        self.defense_active = False
        self.formation_done = False
        self.min_spacing_km = 50  # Minimum spacing between ships in kilometers
        self.repulsion_force = 0.5  # Strength of repulsion between ships
        self.repulsion_threshold_km = 1.0  # Distance at which repulsion starts

    def apply_repulsion(self):
        """Apply repulsion forces between defender ships that are too close."""
        for i, ship1 in enumerate(self.env.defender_ships):
            for j, ship2 in enumerate(self.env.defender_ships[i+1:], i+1):
                # Calculate distance between ships using haversine
                distance = haversine_distance(ship1.lat, ship1.lon, ship2.lat, ship2.lon)
                
                if distance < self.repulsion_threshold_km:
                    # Calculate repulsion vector in lat/lon space
                    dlat = ship2.lat - ship1.lat
                    dlon = ship2.lon - ship1.lon
                    
                    # Normalize the vector
                    magnitude = math.sqrt(dlat**2 + dlon**2)
                    if magnitude > 0:
                        dlat /= magnitude
                        dlon /= magnitude
                    
                    # Calculate repulsion strength (stronger at closer distances)
                    force = self.repulsion_force * (1 - distance / self.repulsion_threshold_km)
                    
                    # Apply repulsion to both ships in opposite directions
                    # Convert force to lat/lon degrees (approximately)
                    force_lat = force * dlat / 111  # 111 km per degree of latitude
                    force_lon = force * dlon / (111 * math.cos(math.radians(ship1.lat)))
                    
                    # Move ships apart
                    ship1.lat -= force_lat
                    ship1.lon -= force_lon
                    ship2.lat += force_lat
                    ship2.lon += force_lon
                    
                    # Update pixel positions
                    ship1.update_pixel_position()
                    ship2.update_pixel_position()

    def _calculate_min_radius(self, num_ships):
        """Calculate minimum radius needed to maintain spacing between ships."""
        circumference = num_ships * self.min_spacing_km
        return max(100, circumference / (2 * math.pi))  # At least 100km radius

    def circular_formation(self, radius=100):
        """Generate target positions in a circular formation with proper spacing."""
        center_lat, center_lon = self.env.hvu.lat, self.env.hvu.lon
        num_defenders = len(self.env.defender_ships)
        
        # Calculate minimum radius needed for proper spacing
        min_radius = self._calculate_min_radius(num_defenders)
        radius = max(radius, min_radius)
        
        angle_step = 2 * math.pi / num_defenders
        return [
            (
                center_lat + (radius / 111) * math.cos(i * angle_step),
                center_lon + (radius / (111 * math.cos(math.radians(center_lat)))) * math.sin(i * angle_step)
            )
            for i in range(num_defenders)
        ]

    def move_defenders_in_circle(self, circle_radius=100, speed=0.01):
        center_lat, center_lon = self.env.hvu.lat, self.env.hvu.lon
        for i, ship in enumerate(self.env.defender_ships):
            angle = (self.env.info['step count'] * speed + i * 2 * np.pi / self.env.num_defenders) % (2 * np.pi)
            new_lat = center_lat + (circle_radius / 111) * np.cos(angle)
            new_lon = center_lon + (circle_radius / (111 * np.cos(np.radians(center_lat)))) * np.sin(angle)
            if self.env.info['step count'] == 0:
                ship.lat, ship.lon = new_lat, new_lon
                ship.update_pixel_position()
            else:
                ship.move_ship_to_coordinates((new_lat, new_lon))

    def set_defenders_in_circle(self, circle_radius=100):
        center_lat, center_lon = self.env.hvu.lat, self.env.hvu.lon
        num_defenders = len(self.env.defender_ships)
        angle_step = 2 * np.pi / num_defenders
        for i, ship in enumerate(self.env.defender_ships):
            angle = i * angle_step
            lat = center_lat + (circle_radius / 111) * np.cos(angle)
            lon = center_lon + (circle_radius / (111 * np.cos(np.radians(center_lat)))) * np.sin(angle)
            ship.lat, ship.lon = lat, lon
            ship.update_pixel_position()

    def check_for_defense_activation(self):
        in_range_defenders = self.attacker_within_defender_range()
        if in_range_defenders:
            self.att_revealed_pos = (self.env.attacker_ship.lat, self.env.attacker_ship.lon)
            self.hvu_revealed_pos = (self.env.hvu.lat, self.env.hvu.lon)
            self.defense_active = True

    def attacker_within_defender_range(self):
        in_range_def = []
        attacker_ship = self.env.attacker_ship
        for defender in self.env.defender_ships:
            if defender.target_in_range(attacker_ship):
                in_range_def.append(defender)
        return in_range_def

    def handle_defense_mechanism(self, formation_type):
        if not self.formation_done:
            self.formation_done = self.move_defenders_in_formation(formation_type)
        if self.formation_done:
            self.exit_direction = self.find_escape_direction(
                np.array(self.hvu_revealed_pos), np.array(self.att_revealed_pos)
            )
            is_unit_safe = self.move_unit_away_from_attacker(self.exit_direction)
            if is_unit_safe:
                self.defense_active = False
                self.formation_done = False

    def move_defenders_in_formation(self, formation_type):
        if formation_type == 'circle':
            target_positions = self.circular_formation()
        elif formation_type == 'triangle':
            target_positions = self.triangular_formation()
        elif formation_type == 'line':
            target_positions = self.line_formation()
        elif formation_type == 'wedge':
            target_positions = self.wedge_formation()
        elif formation_type == 'semicircle':
            target_positions = self.half_circular_formation()
        if self.env.info['step count'] == 0:
            for i, ship in enumerate(self.env.defender_ships):
                ship.lat, ship.lon = target_positions[i][0], target_positions[i][1]
                ship.update_pixel_position()
            return True
        else:
            return self.make_formation(self.env.defender_ships, target_positions)



    def half_circular_formation(self, radius=100, start_angle_deg=90):
        center_lat, center_lon = self.env.hvu.lat, self.env.hvu.lon
        num_defenders = len(self.env.defender_ships)
        start_rad = np.radians(start_angle_deg)
        end_rad = start_rad + np.pi
        angle_step = (end_rad - start_rad) / (num_defenders - 1) if num_defenders > 1 else 0
        return [
            (
                center_lat + (radius / 111) * np.cos(start_rad + i * angle_step),
                center_lon + (radius / (111 * np.cos(np.radians(center_lat)))) * np.sin(start_rad + i * angle_step)
            )
            for i in range(num_defenders)
        ]




    # Function to generate target positions in a circular formation
    def triangular_formation(self, radius_increment=0.5, initial_angle=0):
        """
        Creates a triangular formation using lat/lon units (degrees), based on radius in kilometers.
        """
        positions = []
        angle_increment = 360 / 3  # 3 ships per triangle
        num_circles = math.ceil(len(self.env.defender_ships) / 3)
        center_lat, center_lon = self.env.hvu.lat, self.env.hvu.lon

        for circle in range(1, num_circles + 1):
            radius = circle * radius_increment  # in km
            rotation_angle = initial_angle + (circle - 1) * 180
            for i in range(3):
                angle_deg = i * angle_increment + rotation_angle
                angle_rad = np.radians(angle_deg)
                lat = center_lat + (radius * np.cos(angle_rad)) / 111
                lon = center_lon + (radius * np.sin(angle_rad)) / (111 * np.cos(np.radians(center_lat)))
                positions.append((lat, lon))
        return positions



    def line_formation(self, line_length_km=2.5, distance_from_hvu_km=0.3):
        hvu_lat, hvu_lon = self.hvu_revealed_pos
        attacker_lat, attacker_lon = self.att_revealed_pos

    # Compute direction from HVU to Attacker (normalized)
        direction = np.array([attacker_lat - hvu_lat, attacker_lon - hvu_lon], dtype=np.float64)
        direction /= np.linalg.norm(direction)

    # Perpendicular to direction
        perp = np.array([-direction[1], direction[0]])

    # Shift center from HVU
        center_lat = hvu_lat + (distance_from_hvu_km / 111) * direction[0]
        center_lon = hvu_lon + (distance_from_hvu_km / (111 * np.cos(np.radians(hvu_lat)))) * direction[1]

        return self.find_points_along_vector((center_lat, center_lon), perp, line_length_km)


    def find_points_along_vector(self, center, direction_vector, line_length_km):
        """
        Calculate n equally spaced geographic positions (lat/lon) along a vector centered at 'center'.
        Uses haversine-scaled deltas.

        Args:
            center: tuple (lat, lon)
            direction_vector: numpy array [dx, dy] representing direction
            line_length_km: total length of the line in kilometers
        """
        total_points = len(self.env.defender_ships)

    # Normalize the direction vector
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Spacing between ships in kilometers
        if total_points == 1:
            spacing = 0
        else:
            spacing = line_length_km / (total_points - 1)

        lat_center, lon_center = center
        positions = []

        for i in range(total_points):
            offset_km = (i - (total_points - 1) / 2) * spacing  # center-balanced
            delta_lat = (offset_km * direction_vector[0]) / 111  # 1 deg lat ≈ 111 km
            delta_lon = (offset_km * direction_vector[1]) / (111 * np.cos(np.radians(lat_center)))
            lat = lat_center + delta_lat
            lon = lon_center + delta_lon
            positions.append((lat, lon))

        return positions


    def wedge_formation(self, spread_km=300, distance_from_hvu_km=40):
        """
        Generates geographic positions for n defender ships in a V-shaped (wedge) formation centered around the HVU.
        """
        hvu_lat, hvu_lon = self.hvu_revealed_pos
        att_lat, att_lon = self.att_revealed_pos

    # Convert to vector in lat/lon space (not pixels!)
        hvu_vec = np.array([hvu_lat, hvu_lon])
        att_vec = np.array([att_lat, att_lon])
        direction_vector = att_vec - hvu_vec
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Approximate unit degree deltas
        dlat_unit = direction_vector[0]
        dlon_unit = direction_vector[1]

    # Compute center of V formation in lat/lon
        center_lat = hvu_lat + (distance_from_hvu_km * dlat_unit) / 111
        center_lon = hvu_lon + (distance_from_hvu_km * dlon_unit) / (111 * np.cos(np.radians(hvu_lat)))

        perp_vector = np.array([-dlon_unit, dlat_unit])
        n = len(self.env.defender_ships)
        positions = []

        for i in range(n):
            x_offset_km = (i - (n // 2)) * spread_km / n
            y_offset_km = abs(i - (n // 2)) * (spread_km / n)

        # Geographic offsets
            delta_lat = (x_offset_km * perp_vector[0] - y_offset_km * dlat_unit) / 111
            delta_lon = (x_offset_km * perp_vector[1] - y_offset_km * dlon_unit) / (111 * np.cos(np.radians(center_lat)))

            lat = center_lat + delta_lat
            lon = center_lon + delta_lon
            positions.append((lat, lon))

        return positions

    def make_formation(self, defenders, target_positions):
        """
        Move each defender to their assigned geographic (lat, lon) target position.
        """
        formation_flags = [
            ship.move_ship_to_coordinates((lat, lon))
            for ship, (lat, lon) in zip(defenders, target_positions)
        ]
        return all(formation_flags)



    def find_escape_direction(self, hvu_position, attacker_position):
        """
        Compute unit direction vector from attacker to HVU in lat/lon space (returns numpy array).
        """
        attacker_vec = np.array(attacker_position)
        hvu_vec = np.array(hvu_position)
        escape_vec = hvu_vec - attacker_vec
        return escape_vec / np.linalg.norm(escape_vec)



    def move_unit_away_from_attacker(self, direction, safe_distance=300):
        """
        Move all defenders and HVU geographically in the opposite direction from the attacker.
        Return True if HVU is in a safe zone (≥ safe_distance km).
        """
    # Move defenders
        for ship in self.env.defender_ships:
            delta_lat = ship.speed * direction[0] / 111
            delta_lon = ship.speed * direction[1] / (111 * math.cos(math.radians(ship.lat)))
            ship.lat += delta_lat
            ship.lon += delta_lon
            ship.update_pixel_position()

    # Move HVU
        delta_lat = self.env.hvu.speed * direction[0] / 111
        delta_lon = self.env.hvu.speed * direction[1] / (111 * math.cos(math.radians(self.env.hvu.lat)))
        self.env.hvu.lat += delta_lat
        self.env.hvu.lon += delta_lon
        self.env.hvu.update_pixel_position()

    # Check haversine distance to attacker
        attacker_lat, attacker_lon = self.att_revealed_pos
        hvu_lat, hvu_lon = self.env.hvu.lat, self.env.hvu.lon
        dist = haversine_distance(hvu_lat, hvu_lon, attacker_lat, attacker_lon)
        return dist >= safe_distance

class FireMechanism:
    def __init__(self, environment):
        self.env = environment

    def _handle_firing_mechanics(self):
        reward = 0

    # ✅ Attacker firing at HVU
        self.env.attacker_ship.update_pixel_position()
        self.env.hvu.update_pixel_position()

        if self.env.attacker_ship.target_in_range(self.env.hvu):
            reward += 2  # Reward for HVU being in attacker's firing range

            if self.env.check_los_attacker():
                reward += 5  # Reward for having line-of-sight
                reward += self._handle_attacker_firing()
                self.env.attacker_fired = True  # HVU will now start escaping

    # ✅ Defender firing at attacker
        for defender in self.env.defender_ships:
            defender.update_pixel_position()
            self.env.attacker_ship.update_pixel_position()

            if defender.target_in_range(self.env.attacker_ship):
                reward -= 3  # Penalty for attacker being in defender's range

                if self.env.check_los_defender(defender):
                    reward -= 5  # Penalty for attacker being in defender's LOS
                    reward -= self._handle_defender_firing(defender)

        return reward


    def _handle_attacker_firing(self):
        reward = 0
        if self.validate_and_fire(self.env.attacker_ship, self.env.hvu):
            self.env.attacker_fired = True  # ✅ Ensure defender response
            reward += 20  # ✅ Reward only if fire actually happened
        return reward



    def _handle_defender_firing(self, defender):
        penalty = 0
        # Defender only fires when attacker is firing
        if self.env.attacker_fired:
            if self.validate_and_fire(defender, self.env.attacker_ship): # check and Fire the torpedo towards target
                penalty += 20
        return penalty


    def validate_and_fire(self, ship, target):
        """Fires a torpedo at the target if within range and cooldown periods are satisfied."""
        current_time = time.time()
        torpedo_fired = False

        if current_time - ship.last_fire_time >= ship.reload_delay:  # Check if ship can fire
            if ship.target_lock_time == 0:  # Locking target for the first time
                ship.target_lock_time = current_time

            if current_time - ship.target_lock_time >= ship.target_delay:  # Fire if lock complete
                torpedo_fired = self.fire_torpedo(ship, target)

        return torpedo_fired

    def fire_torpedo(self, ship, target):
        """Fire a torpedo toward the target using lat/lon direction and storage."""
        if ship.torpedo_count > 0:
        # ✅ Compute geographic direction (bearing)
            delta_lon = math.radians(target.lon - ship.lon)
            delta_lat = math.radians(target.lat - ship.lat)
            direction = math.atan2(delta_lat, delta_lon)

        # ✅ Create torpedo using lat/lon instead of x/y
            torpedo = Torpedo(
                torpedo_id=f"T{ship.ship_id}_{ship.torpedo_count}",
                lat=ship.lat,
                lon=ship.lon,
                speed=ship.torpedo_fire_speed,
                damage=ship.torpedo_damage,
                direction=direction,
                source=ship,
                target=target,
                mapGenerator=self.env.mapGenerator  # Needed for pixel conversion at render time
            )

            ship.torpedoes.append(torpedo)
            ship.torpedo_count -= 1
            ship.last_fire_time = time.time()
            ship.target_lock_time = 0
            return True

        return False



    def _update_torpedo_position(self):
        """Update the torpedo positions and calculate rewards/penalties for the attacker and defenders."""
        # Update attacker torpedo positions and calculate rewards
        attacker_reward, att_target_destroyed = self.update_torpedo(self.env.attacker_ship, self.env.ships)
        self.env.reward += attacker_reward

        # Check if the attacker destroyed the HVU
        if att_target_destroyed:
            self.env.info['HVU destroyed'] += 1
            self.env.done = True
            return

        # Update defender torpedo positions and calculate penalties
        for defender in self.env.defender_ships:
            penalty, def_target_destroyed = self.update_torpedo(defender, self.env.ships)
            self.env.reward -= penalty

            # Check if any defender destroyed the attacker
            if def_target_destroyed:
                self.env.info['attacker destroyed'] += 1
                self.env.done = True
                return


    def update_torpedo(self, ship, env_ships, threshold=10):
        """Update torpedo positions, check for collisions, and determine rewards."""
        reward = 0
        target_destroyed = False

        for torpedo in ship.torpedoes[:]:
            torpedo.move()

            # Remove torpedoes that leave the screen or hit a target
            if not torpedo.within_bounds(self.env.width, self.env.height) or torpedo.check_collision(env_ships, threshold):
                ship.torpedoes.remove(torpedo)
                continue

            # Handle hitting the target
            r, target_destroyed = torpedo.hit_target(threshold)
            reward += r
            if torpedo.target_hit:
                ship.torpedoes.remove(torpedo)

            if target_destroyed:  # Stop further processing if target is destroyed
                break

        return reward, target_destroyed



    def check_collisions(self):
        """Check for collisions between attacker and HVU or defenders."""
        # Check for collision between attacker and the HVU
        if self.check_collision((self.env.attacker_ship.lat, self.env.attacker_ship.lon),
                              (self.env.hvu.lat, self.env.hvu.lon)):
            print(f"Collision detected between the attacker and the HVU! Both ships are destroyed.")
            self.env.attacker_ship.ship_health = 0
            self.env.hvu.ship_health = 0
            return True

        # Check for collision between attacker and any of the defender ships
        for defender_ship in self.env.defender_ships:
            if self.check_collision((self.env.attacker_ship.lat, self.env.attacker_ship.lon),
                                 (defender_ship.lat, defender_ship.lon)):
                print(f"Collision detected between the attacker and defender {defender_ship.ship_id}! Both ships are destroyed.")
                self.env.attacker_ship.ship_health = 0
                defender_ship.ship_health = 0
                return True

        return False

    def check_collision(self, pos1, pos2, collision_range=0.5):
        """
        Check if two entities are within collision range using haversine distance.
        collision_range is in kilometers.
        """
        return haversine_distance(pos1[0], pos1[1], pos2[0], pos2[1]) < collision_range

import time
import math
import numpy as np

class DecoyMissileManager:
    def __init__(self, env):
        self.env = env

    def handle_DecoyM_firing_mechanics(self):
        if self.env.attacker_ship.torpedoes:
            for active_torpedo in self.env.attacker_ship.torpedoes:
                torpedo_px = self.env.mapGenerator._latlon_to_pixels(active_torpedo.lat, active_torpedo.lon)

                for defender in self.env.defender_ships:
                    if defender.ship_type == 'def_decoyM':
                        defender_px = self.env.mapGenerator._latlon_to_pixels(defender.lat, defender.lon)
                        distance = np.linalg.norm(np.array(defender_px) - np.array(torpedo_px))

                        if distance <= defender.firing_range:  # ✅ now comparing pixels vs pixels
                            self.validate_and_fire(defender, active_torpedo)


    def validate_and_fire(self, ship, target):
        current_time = time.time()
        if current_time - ship.last_decoy_fire_time >= ship.reload_delay:
            if ship.decoy_target_lock_time == 0:
                ship.decoy_target_lock_time = current_time

            if current_time - ship.decoy_target_lock_time >= ship.target_delay:
                return self.fire_missile(ship, target)
        return False

    def fire_missile(self, ship, target):
        if ship.decoyM_count > 0:
            # Compute direction (bearing) from ship to target in radians
            delta_lat = math.radians(target.lat - ship.lat)
            delta_lon = math.radians(target.lon - ship.lon)
            direction = math.atan2(delta_lat, delta_lon)

            missile = {
                'id': f"DM{ship.ship_id}_{ship.decoyM_count}",
                'lat': ship.lat,
                'lon': ship.lon,
                'direction': direction,
                'speed': ship.decoyM_speed,  # In km/frame
                'source': ship,
                'target': target,
                'target_hit': False
            }

            ship.decoy_missile.append(missile)
            ship.decoyM_count -= 1
            ship.last_decoy_fire_time = time.time()
            ship.decoy_target_lock_time = 0
            return True
        return False

    def update_decoy_missile(self):
        for ship in self.env.defender_ships:
            if ship.ship_type != 'def_decoyM':
                continue

            missiles_to_remove = []
            for missile in ship.decoy_missile:
                # Move the missile in geographic space
                lat = missile['lat']
                lon = missile['lon']
                direction = missile['direction']
                speed = missile['speed']

                delta_lat = speed * math.cos(direction) / 111
                delta_lon = speed * math.sin(direction) / (111 * math.cos(math.radians(lat)))

                missile['lat'] += delta_lat
                missile['lon'] += delta_lon

                # Convert to pixel position for collision and rendering
                missile['x'], missile['y'] = self.env.mapGenerator._latlon_to_pixels(missile['lat'], missile['lon'])

                target = missile['target']
                if target not in self.env.attacker_ship.torpedoes:
                    missiles_to_remove.append(missile)
                    continue

                target_x, target_y = self.env.mapGenerator._latlon_to_pixels(target.lat, target.lon)
                missile_pos = np.array([missile['x'], missile['y']])
                target_pos = np.array([target_x, target_y])
                distance = np.linalg.norm(target_pos - missile_pos)

                if distance < ship.decoyM_blast_range:
                    if target in target.source.torpedoes:
                        target.source.torpedoes.remove(target)
                    missiles_to_remove.append(missile)
                    missile['target_hit'] = True
                    continue

                if not self.within_bounds(missile) or self.check_collision(self.env.ships, missile, ship.decoyM_blast_range):
                    missiles_to_remove.append(missile)

            for missile in missiles_to_remove:
                if missile in ship.decoy_missile:
                    ship.decoy_missile.remove(missile)

    def within_bounds(self, missile):
        return (
            0 <= float(missile['x']) <= self.env.width and
            0 <= float(missile['y']) <= self.env.height
        )

    def check_collision(self, ships, missile, threshold=0.5):
        """
        Check for collisions between a decoy missile and ships.
        Uses haversine distance for geographic collision detection.
        Threshold is in kilometers.
        """
        for ship in ships:
            if ship.ship_id == missile['source'].ship_id:
                continue

            distance = haversine_distance(missile['lat'], missile['lon'], ship.lat, ship.lon)
            if distance < threshold:
                return True
        return False

import numpy as np
import math
import time

class HelicopterManager:
    """
    Manages helicopters using zoom-invariant lat/lon positions.
    Each helicopter:
    - Takes off from a defender ship
    - Moves to a circular entry point around the HVU
    - Circles the HVU
    - Returns to the original defender
    - Lands and increments helicop_count
    - Moves to next defender
    """
    def __init__(self, env):
        self.env = env
        self.helicopter_lat = None
        self.helicopter_lon = None
        self.helicopter_index = 0  # Current defender index
        self.helicopter_active = False
        self.helicopter_state = None  # States: None, "takeoff", "circle", "return", "landing"
        self.helicopter_x = 0  # Pixel coordinates for rendering
        self.helicopter_y = 0  # Pixel coordinates for rendering
        self.current_angle = 0
        self.initial_circle_angle = None
        self.completed_degrees = 0
        self.helicopter_angle = 0
        self.original_defender = None  # Keep track of which defender launched the helicopter
        self.landing_threshold = 0.001  # Degrees lat/lon for landing detection
        self.circle_completed = False
        self.entry_point = None
        self.landing_start_time = None
        self.landing_duration = 2.0  # Seconds for landing animation

    def update_pixel_position(self):
        """Convert helicopter's lat/lon coordinates to screen pixel coordinates."""
        if self.helicopter_lat is not None and self.helicopter_lon is not None:
            self.helicopter_x, self.helicopter_y = self.env.mapGenerator._latlon_to_pixels(
                self.helicopter_lat,
                self.helicopter_lon
            )
            self.helicopter_x = np.clip(self.helicopter_x, 0, self.env.width - 1)
            self.helicopter_y = np.clip(self.helicopter_y, 0, self.env.height - 1)

    def move_defenders_helicop(self):
        """Main update function for helicopter movement."""
        if not self.helicopter_active and len(self.env.defender_ships) > 0:
            self._try_launch_helicopter()
            return

        if not self.helicopter_active:
            return

        # Get current positions
        hvu_pos = (self.env.hvu.lat, self.env.hvu.lon)
        defender_pos = (self.original_defender.lat, self.original_defender.lon)

        # State machine for helicopter movement
        if self.helicopter_state == "takeoff":
            if not self.entry_point:
                self.entry_point = self._calculate_entry_point(hvu_pos)
            if self._move_to_point(self.entry_point):
                self._transition_to_circle()

        elif self.helicopter_state == "circle":
            if self._circle_around_point(hvu_pos):
                self.helicopter_state = "return"

        elif self.helicopter_state == "return":
            if self._move_to_point(defender_pos):
                self.helicopter_state = "landing"
                self.landing_start_time = time.time()

        elif self.helicopter_state == "landing":
            if self._handle_landing():
                self._complete_mission()

    def _try_launch_helicopter(self):
        """Try to launch a helicopter from the current defender."""
        attempts = 0
        while attempts < len(self.env.defender_ships):
            defender = self.env.defender_ships[self.helicopter_index]
            if defender.ship_type == 'def_heli' and defender.helicop_count > 0:
                self._launch_helicopter(defender)
                break
            self.helicopter_index = (self.helicopter_index + 1) % len(self.env.defender_ships)
            attempts += 1

    def _launch_helicopter(self, defender):
        """Launch a helicopter from the given defender."""
        defender.helicop_count -= 1
        self.original_defender = defender
        self.helicopter_lat = defender.lat
        self.helicopter_lon = defender.lon
        self.helicopter_active = True
        self.helicopter_state = "takeoff"
        self.current_angle = 0
        self.initial_circle_angle = None
        self.completed_degrees = 0
        self.circle_completed = False
        self.entry_point = None
        self.landing_start_time = None
        self.update_pixel_position()

    def _transition_to_circle(self):
        """Handle transition to circling state."""
        self.helicopter_state = "circle"
        self.circle_completed = False
        self.completed_degrees = 0
        self.initial_circle_angle = math.degrees(math.atan2(
            self.helicopter_lon - self.env.hvu.lon,
            self.helicopter_lat - self.env.hvu.lat
        ))
        self.current_angle = self.initial_circle_angle

    def _handle_landing(self):
        """Handle landing animation and state. Returns True when landing complete."""
        if self.landing_start_time is None:
            self.landing_start_time = time.time()
            return False

        elapsed = time.time() - self.landing_start_time
        if elapsed >= self.landing_duration:
            return True

        # Smooth landing animation
        progress = min(1.0, elapsed / self.landing_duration)
        target_lat = self.original_defender.lat
        target_lon = self.original_defender.lon

        self.helicopter_lat = self.helicopter_lat + (target_lat - self.helicopter_lat) * progress
        self.helicopter_lon = self.helicopter_lon + (target_lon - self.helicopter_lon) * progress
        self.update_pixel_position()

        return False

    def _calculate_entry_point(self, hvu_pos):
        """Calculate nearest point on circle around HVU based on current position."""
        hvu_lat, hvu_lon = hvu_pos

        # Calculate direction vector from HVU to helicopter
        delta_lat = self.helicopter_lat - hvu_lat
        delta_lon = self.helicopter_lon - hvu_lon

        if delta_lat == 0 and delta_lon == 0:
            # If directly at HVU, choose arbitrary angle
            angle = 0
        else:
            # Calculate angle from HVU to helicopter
            angle = math.atan2(delta_lon, delta_lat)

        # Calculate entry point on circle using haversine radius
        radius_km = self.env.helicop_path_radius
        radius_deg = radius_km / 111.0  # Convert km to degrees (approximate)

        # Apply latitude compensation for longitude distances
        lat_factor = math.cos(math.radians(hvu_lat))

        # Calculate entry point coordinates
        entry_lat = hvu_lat + (radius_deg * math.cos(angle))
        entry_lon = hvu_lon + (radius_deg * math.sin(angle) / lat_factor)

        return (entry_lat, entry_lon)

    def _move_to_point(self, target_point):
        """Move helicopter toward target point, returns True if arrived."""
        if not target_point or self.helicopter_lat is None:
            return False

        target_lat, target_lon = target_point

        # Calculate distance using haversine
        distance = haversine_distance(
            self.helicopter_lat, self.helicopter_lon,
            target_lat, target_lon
        )

        # Update helicopter angle for rendering
        bearing = math.degrees(math.atan2(
            target_lon - self.helicopter_lon,
            target_lat - self.helicopter_lat
        ))
        self.helicopter_angle = bearing

        # Check if we've arrived
        if distance <= self.landing_threshold:
            self.helicopter_lat = target_lat
            self.helicopter_lon = target_lon
            self.update_pixel_position()
            return True

        # Speed scales with zoom level
        speed_km = self.env.helicop_speed * self._get_zoom_scale()
        move_ratio = min(speed_km / distance, 1.0) if distance > 0 else 0

        # Calculate new position
        delta_lat = target_lat - self.helicopter_lat
        delta_lon = target_lon - self.helicopter_lon

        self.helicopter_lat += delta_lat * move_ratio
        self.helicopter_lon += delta_lon * move_ratio

        self.update_pixel_position()
        return False

    def _circle_around_point(self, center_point):
        """Circle around point, returns True when full circle complete."""
        center_lat, center_lon = center_point

        # Scale radius and angular speed based on zoom
        radius_km = self.env.helicop_path_radius
        angular_speed = 2.0 * self._get_zoom_scale()

        # Update angle and track progress
        self.current_angle = (self.current_angle + angular_speed) % 360
        self.completed_degrees += angular_speed

        # Set helicopter angle tangent to circle
        self.helicopter_angle = (self.current_angle + 90) % 360

        # Convert radius to degrees (approximate)
        radius_deg = radius_km / 111.0

        # Calculate new position with latitude compensation
        lat_factor = math.cos(math.radians(center_lat))
        angle_rad = math.radians(self.current_angle)

        self.helicopter_lat = center_lat + radius_deg * math.cos(angle_rad)
        self.helicopter_lon = center_lon + (radius_deg * math.sin(angle_rad) / lat_factor)

        self.update_pixel_position()
        return self.completed_degrees >= 360

    def _complete_mission(self):
        """Complete helicopter mission and reset for next one."""
        if self.original_defender:
            self.original_defender.helicop_count += 1  # Return helicopter to defender

        # Reset all state
        self.helicopter_active = False
        self.helicopter_state = None
        self.helicopter_lat = None
        self.helicopter_lon = None
        self.current_angle = 0
        self.completed_degrees = 0
        self.initial_circle_angle = None
        self.circle_completed = False
        self.entry_point = None
        self.original_defender = None

        # Move to next defender
        self.helicopter_index = (self.helicopter_index + 1) % len(self.env.defender_ships)

    def _get_zoom_scale(self):
        """Calculate movement scale factor based on zoom level."""
        base_zoom = 6.0
        current_zoom = self.env.zoom
        if current_zoom < base_zoom:
            return max(0.5, (base_zoom / current_zoom) * 0.5)
        else:
            return min(2.0, (current_zoom / base_zoom) * 0.5)

import requests
class MapGenerator:

    def __init__(self, env, map_center=[3.0000, 86.0000], zoom=6):

        self.env = env
        self.map_center = map_center  # Store center lat/lon
        self.zoom = zoom  # Store zoom level
        self._calculate_bbox_from_center_and_zoom()  # Calculate bounding box
        self.map_initialized = False  # Flag to check if the map has been initialized
        self.map_image_path = None
        self.map_image = None

        # Initialize the map once
        self.map_initialise()

    def _calculate_bbox_from_center_and_zoom(self):
        """Calculate bounding box symmetrically around the fixed map center."""
        center_lat, center_lon = self.map_center

    # Degrees per zoom factor (smaller as zoom increases)
        zoom_factor = 180 / (2 ** (self.zoom - 1))
        half_lat_span = zoom_factor / 2
        half_lon_span = zoom_factor  # Wider to match typical 2:1 aspect ratio

    # Compute symmetrical bbox
        min_lat = max(-85, center_lat - half_lat_span)
        max_lat = min(85, center_lat + half_lat_span)
        min_lon = max(-180, center_lon - half_lon_span)
        max_lon = min(180, center_lon + half_lon_span)

        return (min_lon, min_lat, max_lon, max_lat)


    def render_background(self):
        """Renders the real-world map as the background with accurate grid overlay."""
        self.env.screen.blit(self.map_image, (0, 0))  # Use preloaded map
        self._draw_grid(self.env.screen)  # Draw grid over the map

    def map_initialise(self):
        """Loads the real-world map and scales it for display."""
        if self.map_initialized:
            return  # Prevent re-initialization if already done

        self.map_image_path = self.generate_map()
        if not os.path.exists(self.map_image_path):
            raise FileNotFoundError("Error: The generated map image was not found!")

        self.map_image = pygame.image.load(self.map_image_path)
        self.map_image = pygame.transform.scale(self.map_image, (self.env.width, self.env.height))
        self.map_initialized = True  # Mark initialization complete

#     def _generate_map(self):
#         """Generates a real-world map with real latitude-longitude grid lines."""
#         html_map_file = "map.html"
#         png_map_file = "map.png"

#         # Generate Folium map
#         folium_map = folium.Map(location=self.map_center, zoom_start=self.zoom)

#         # Add real-world latitude/longitude grid lines
# #         lat_spacing, lon_spacing = self._get_grid_spacing(self.zoom)

# #         for lat in np.arange(-90, 90, lat_spacing):
# #             folium.PolyLine([(lat, -180), (lat, 180)], color="blue", weight=0.5).add_to(folium_map)

# #         for lon in np.arange(-180, 180, lon_spacing):
# #             folium.PolyLine([(-90, lon), (90, lon)], color="blue", weight=0.5).add_to(folium_map)

#         # ✅ Add latitude & longitude popups to verify accuracy
#         folium_map.add_child(folium.LatLngPopup())

#         folium_map.save(html_map_file)

#         # Convert HTML map to PNG using Selenium
#         options = webdriver.ChromeOptions()
#         options.add_argument("--headless")
#         driver = webdriver.Chrome(options=options)

#         driver.get(f"file://{os.path.abspath(html_map_file)}")
#         driver.set_window_size(env.width, env.height)
#         driver.save_screenshot(png_map_file)
#         driver.quit()

#         return png_map_file




    def generate_map(self):
        wms_url = "http://localhost:8080/geoserver/ne/wms"

    # --- Compute bbox based on zoom ---

        # Use fixed bbox
        self.bbox = self._calculate_bbox_from_center_and_zoom()
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = self.bbox
        bbox = f"{self.min_lon},{self.min_lat},{self.max_lon},{self.max_lat}"

# Degrees-per-pixel scaling
        if self.env.width == 0 or self.env.height == 0:
            raise ValueError("Screen dimensions cannot be zero.")

        self.degrees_per_pixel_lon = (self.max_lon - self.min_lon) / self.env.width
        self.degrees_per_pixel_lat = (self.max_lat - self.min_lat) / self.env.height

    # --- Save for use in movement ---

        params = {
            "service": "WMS",
            "version": "1.1.1",
            "request": "GetMap",
            "layers": "ne:world",
            "styles": "",
            "format": "image/png",
            "transparent": "true",
            "srs": "EPSG:4326",
            "bbox": bbox,
            "width": self.env.width,
            "height": self.env.height
        }

        response = requests.get(wms_url, params=params)
        if response.status_code == 200:
            with open("map.png", "wb") as f:
                f.write(response.content)
            return "map.png"
        else:
            print("GeoServer WMS fetch failed.")
            print(response.text)
            raise Exception("GeoServer WMS fetch failed with status:", response.status_code)


    def _get_grid_spacing(self, zoom):
        """Determines latitude/longitude grid spacing based on zoom level."""
        if zoom >= 12:
            return 0.01, 0.01
        elif zoom >= 10:
            return 0.1, 0.1
        elif zoom >= 8:
            return 0.5, 0.5
        elif zoom >= 6:
            return 1, 1
        elif zoom >= 4:
            return 5, 5
        else:
            return 10, 10

    # def _latlon_to_pixels(self, lat, lon):
    #     """Convert lat/lon to screen x, y using Mercator projection and current bbox."""
    # # Clamp latitude to Mercator limit
    #     lat = max(min(lat, 85.0), -85.0)

    # # Normalized Mercator Y
    #     lat_rad = math.radians(lat)
    #     merc_n = math.log(math.tan(math.pi / 4 + lat_rad / 2))

    #     x = (lon - self.min_lon) / (self.max_lon - self.min_lon) * self.env.width
    #     y = (1 - (merc_n - math.log(math.tan(math.pi / 4 + math.radians(self.min_lat) / 2))) /
    #          (math.log(math.tan(math.pi / 4 + math.radians(self.max_lat) / 2)) -
    #           math.log(math.tan(math.pi / 4 + math.radians(self.min_lat) / 2)))) * self.env.height

    #     return int(x), int(y)
    def _latlon_to_pixels(self, lat, lon):
        """Convert lat/lon to screen pixels using Mercator projection."""
    # Clamp latitude to Mercator bounds
        MAX_LAT = 85.05113
        lat = max(min(lat, MAX_LAT), -MAX_LAT)

    # Convert degrees to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

    # Mercator projection formulas
        merc_n = math.log(math.tan(math.pi / 4 + lat_rad / 2))
        min_merc = math.log(math.tan(math.pi / 4 + math.radians(self.min_lat) / 2))
        max_merc = math.log(math.tan(math.pi / 4 + math.radians(self.max_lat) / 2))

        x = (lon - self.min_lon) / (self.max_lon - self.min_lon) * self.env.width
        y = (1 - (merc_n - min_merc) / (max_merc - min_merc)) * self.env.height

        return int(x), int(y)


        # Map dimensions


    def _draw_grid(self, screen, grid_color=(200, 200, 200)):
        """
        Draws real-world latitude & longitude grid lines with labeled coordinates.
        """

        # Determine grid spacing dynamically based on zoom level
        lat_spacing, lon_spacing = self._get_grid_spacing(self.zoom)

        # Screen dimensions
        screen_width, screen_height = self.env.width, self.env.height

        # Store grid labels to avoid overlapping
        grid_labels = []

        # Draw latitude lines (horizontal)
        for lat in np.arange(-90, 90, lat_spacing):
            start_x, start_y = self._latlon_to_pixels(lat, -180)
            end_x, end_y = self._latlon_to_pixels(lat, 180)

            pygame.draw.line(screen, grid_color, (start_x, start_y), (end_x, end_y), 1)

            # Add latitude label at the left side of the map
            if 0 <= start_y < screen_height:
                grid_labels.append((10, start_y - 5, f"{lat:.1f}°"))

        # Draw longitude lines (vertical)
        for lon in np.arange(-180, 180, lon_spacing):
            start_x, start_y = self._latlon_to_pixels(-90, lon)
            end_x, end_y = self._latlon_to_pixels(90, lon)

            pygame.draw.line(screen, grid_color, (start_x, start_y), (end_x, end_y), 1)

            # Add longitude label at the top of the map
            if 0 <= start_x < screen_width:
                grid_labels.append((start_x + 5, 10, f"{lon:.1f}°"))

        # Render latitude & longitude labels
        font = pygame.font.SysFont('Arial', 12, bold=True)
        for x, y, text in grid_labels:
            text_surface = font.render(text, True, (255, 255, 255))  # White text
            screen.blit(text_surface, (x, y))


    def pixels_to_latlon(self, x, y):
        """Converts pixel (x, y) coordinates back to latitude & longitude using Mercator projection."""

        # Convert latitude from Mercator projection back to degrees
        def mercator_to_lat(merc_y):
            return math.degrees(2 * math.atan(math.exp(merc_y)) - math.pi / 2)

        # Google Maps uses linear scaling for longitude
        def mercator_to_lon(merc_x):
            return merc_x * 180.0  # Convert back to longitude range [-180, 180]

        # Map dimensions
        screen_width, screen_height = self.env.width, self.env.height

        # Scale factor for given zoom level
        scale = 256 * 2**self.zoom

        # Convert map center to Mercator projection
        center_x = self.map_center[1] / 180.0  # Linear longitude conversion
        center_y = math.log(math.tan((math.pi / 4) + (math.radians(self.map_center[0]) / 2)))

        # Convert pixel coordinates back to Mercator projection
        merc_x = center_x + (x - screen_width / 2) / scale
        merc_y = center_y - (y - screen_height / 2) / scale

        # Convert Mercator values back to lat/lon
        lat = mercator_to_lat(merc_y)
        lon = mercator_to_lon(merc_x)

        return lat, lon


    def _render_latlon(self, x, y, rect_size):
        """
        Render latitude and longitude below the ship at (x, y) position.
        """
        # Convert pixel coordinates to latitude/longitude
        lat, lon = self.pixels_to_latlon(x, y)

        # Format lat/lon to 2 decimal places
        lat_lon_text = f"{lat:.2f}, {lon:.2f}"

        # Render latitude & longitude text
        font = pygame.font.SysFont('Arial', 10, bold=True)
        text_surface = font.render(lat_lon_text, True, (0, 0, 0))  # Black text

        # Get text width & height for centering
        text_width, text_height = text_surface.get_size()

        # Position text **centered below** the ship
        text_x = int(x) - text_width // 2  # Center horizontally
        text_y = int(y) + rect_size // 2 + 5  # Just below the ship, with padding

        # Blit the text onto the screen
        self.env.screen.blit(text_surface, (text_x, text_y))

class UIManager:
    def __init__(self, env):
        self.env = env
        self.title_font = pygame.font.SysFont('Arial', 26, bold=True)
        self.label_font = pygame.font.SysFont('Arial', 18)
        self.input_font = pygame.font.SysFont('Arial', 18)





    def get_user_input_screen(self):
        """
            Displays the user input screen and returns defender setup configuration.
            Returns:
                num_defenders (int): Total number of defender ships
                use_custom (bool): Whether user customized ship configuration
                num_sonar (int): Ships with sonar
                num_heli (int): Ships with helicopter
                num_decoy (int): Ships with decoy
                selected_formation (str): "triangle" or "circle"
            """

        user_inputs, input_boxes, colors = self._initialize_inputs()

        checkbox_checked = False
        error_message = ""
        done = False

        while not done:
            self.env.screen.fill((240, 240, 240))
            self._render_title("Environment Details")

            self._render_checkbox(checkbox_checked)
            self._render_formation_selection(user_inputs)

            submit_button = self._render_inputs(user_inputs, input_boxes, colors, checkbox_checked)
            #self._render_zoom_controls(user_inputs)

            #self._render_zoom_controls(user_inputs)

            if error_message:
                self._render_error(error_message, submit_button.y + 50)

            pygame.display.flip()
            pygame.time.Clock().tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    checkbox_checked, input_boxes, colors = self._handle_mouse_click(
                        event.pos, checkbox_checked, input_boxes
                    )
                    self._handle_formation_click(event.pos, user_inputs)
                    if hasattr(self, 'zoom_in_button') and self.zoom_in_button.collidepoint(event.pos):
                        current_zoom = int(user_inputs["zoom"])
                        if current_zoom < 12:
                            user_inputs["zoom"] = str(current_zoom + 1)

                    elif hasattr(self, 'zoom_out_button') and self.zoom_out_button.collidepoint(event.pos):
                        current_zoom = int(user_inputs["zoom"])
                        if current_zoom > 1:
                            user_inputs["zoom"] = str(current_zoom - 1)

        # Handle submit button
                    if submit_button.collidepoint(event.pos):
                        done, error_message = self._validate_inputs(user_inputs, checkbox_checked)

                elif event.type == pygame.KEYDOWN:
                    user_inputs, colors = self._handle_key_input(event, user_inputs, input_boxes, colors)

        return self._finalize_user_inputs(user_inputs, checkbox_checked)








    def _initialize_inputs(self):
        user_inputs = {
            "total_num_def": str(self.env.num_defenders),
            "num_sonar_def": str(self.env.num_def_with_sonar),
            "num_heli_def": str(self.env.num_def_with_helicopter),
            "num_decoy_def": str(self.env.num_def_with_decoy),
            "num_default_def": str(self.env.num_default_def),
            "map_center": f"{self.env.map_center[0]}, {self.env.map_center[1]}",
            "zoom": str(self.env.zoom),
            "base_location": f"{self.env.base_location[0]}, {self.env.base_location[1]}",
            "def_default_formation": self.env.def_default_formation,
            "def_moving_formation": self.env.def_moving_formation
        }

        input_boxes = {
            "total_num_def": pygame.Rect(100, 120, 180, 35),
            "num_sonar_def": pygame.Rect(100, 390, 180, 35),
            "num_heli_def": pygame.Rect(100, 460, 180, 35),
            "num_decoy_def": pygame.Rect(380, 390, 180, 35),
            "num_default_def": pygame.Rect(380, 460, 180, 35),
            "map_center": pygame.Rect(100, 200, 220, 35),
            "zoom": pygame.Rect(380, 200, 220, 35),
            "base_location": pygame.Rect(380, 120, 220, 35)
        }

        colors = {key: pygame.Color('lightskyblue3') for key in user_inputs}
        return user_inputs, input_boxes, colors


    def _render_title(self, text):
        instructions = self.title_font.render(text, True, (0, 0, 0))  # Render black text
        self.env.screen.blit(instructions, (100, 50))  # Draw at fixed top position


    def _render_checkbox(self, checked):
        self.checkbox_rect = pygame.Rect(100, 285, 20, 20)
        pygame.draw.rect(self.env.screen, (0, 0, 0), self.checkbox_rect, 2)  # Draw outer border

        if checked:
            pygame.draw.rect(self.env.screen, (0, 0, 0), self.checkbox_rect.inflate(-8, -8))  # Fill if checked

        # Label beside checkbox
        label = self.label_font.render("Customize Defender Ship(Optional):", True, (0, 0, 0))
        self.env.screen.blit(label, (100, 255))


#     def _render_formation_selection(self, inputs):
#         label = self.label_font.render("Select Defender Default Formation:", True, (0, 0, 0))
#         self.env.screen.blit(label, (700, 95))

#         start_x = 700
#         start_y = 130
#         gap_y = 40

#         self.formation_rects = []

#         for i, formation in enumerate(self.env.avail_def_default_formations):
#             rect = pygame.Rect(start_x, start_y + i * gap_y, 20, 20)
#             self.formation_rects.append((formation, rect))

#             pygame.draw.rect(self.env.screen, (0, 0, 0), rect, 2)

#             if inputs["def_default_formation"] == formation:
#                 pygame.draw.rect(self.env.screen, (0, 0, 0), rect.inflate(-8, -8))

#             self.env.screen.blit(self.label_font.render(formation.capitalize(), True, (0, 0, 0)), (rect.x + 30, rect.y))


    def _render_formation_selection(self, inputs):
        # First render the existing defender default formation
        label = self.label_font.render("Select Defender Default Formation:", True, (0, 0, 0))
        self.env.screen.blit(label, (680, 95))

        start_x = 700
        start_y = 130
        gap_y = 40

        self.formation_rects = []

        for i, formation in enumerate(self.env.avail_def_default_formations):
            rect = pygame.Rect(start_x, start_y + i * gap_y, 20, 20)
            self.formation_rects.append((formation, rect))

            pygame.draw.rect(self.env.screen, (0, 0, 0), rect, 2)

            if inputs["def_default_formation"] == formation:
                pygame.draw.rect(self.env.screen, (0, 0, 0), rect.inflate(-8, -8))

            self.env.screen.blit(self.label_font.render(formation.capitalize(), True, (0, 0, 0)), (rect.x + 30, rect.y))

        # --- Now Render Moving Formation ---
        move_label = self.label_font.render("Select Defender Moving Formation:", True, (0, 0, 0))
        self.env.screen.blit(move_label, (680, start_y + len(self.env.avail_def_default_formations) * gap_y + 20))

        start_y_moving = start_y + len(self.env.avail_def_default_formations) * gap_y + 55

        self.moving_formation_rects = []

        move_formations = ["triangle", "circle", "semicircle", "wedge", "line"]

        for i, formation in enumerate(move_formations):
            rect = pygame.Rect(start_x, start_y_moving + i * gap_y, 20, 20)
            self.moving_formation_rects.append((formation, rect))

            pygame.draw.rect(self.env.screen, (0, 0, 0), rect, 2)

            if inputs["def_moving_formation"] == formation:
                pygame.draw.rect(self.env.screen, (0, 0, 0), rect.inflate(-8, -8))

            self.env.screen.blit(self.label_font.render(formation.capitalize(), True, (0, 0, 0)), (rect.x + 30, rect.y))



    def _render_inputs(self, inputs, boxes, colors, custom_enabled):
        labels = {
            "total_num_def": "Total Defender Ships:",
            "num_sonar_def": "Defenders with Sonar:",
            "num_heli_def": "Defenders with Helicopter:",
            "num_decoy_def": "Defenders with Decoy Missile:",
            "num_default_def": "Default(generic) Defender Ships:",
            "map_center": "Map Center (Lat,Long):",
            "zoom": "Zoom Level:",
            "base_location": "Base Location (Lat,Long):"
        }

        self._draw_input_box(inputs["total_num_def"], boxes["total_num_def"], labels["total_num_def"], colors["total_num_def"])

        if custom_enabled:
            self.env.screen.blit(
                self.title_font.render("Enter Counts For Each Defender Ship:", True, (0, 0, 0)),
                (100, 320)
            )
            for key in ["num_sonar_def", "num_heli_def", "num_decoy_def", "num_default_def"]:
                self._draw_input_box(inputs[key], boxes[key], labels[key], colors[key])

        for key in ["map_center", "zoom", "base_location"]:
            self._draw_input_box(inputs[key], boxes[key], labels[key], colors[key])


        submit_rect = pygame.Rect(100, 520 if custom_enabled else 330, 180, 40)
        pygame.draw.rect(self.env.screen, (0, 150, 0), submit_rect)
        text = self.input_font.render("Start Simulation", True, (255, 255, 255))
        self.env.screen.blit(text, (submit_rect.x + 25, submit_rect.y + 10))

        return submit_rect


    def _draw_input_box(self, text, rect, label, color):
        pygame.draw.rect(self.env.screen, color, rect, 2)
        self.env.screen.blit(self.label_font.render(label, True, (0, 0, 0)), (rect.x, rect.y - 25))
        self.env.screen.blit(self.input_font.render(text, True, (0, 0, 0)), (rect.x + 5, rect.y + 5))


    def _render_error(self, message, y_pos):
        font = pygame.font.SysFont('Arial', 16)
        error_surface = font.render(message, True, (200, 0, 0))
        self.env.screen.blit(error_surface, (100, y_pos))


    def _handle_mouse_click(self, pos, checkbox_state, input_boxes):
        if self.checkbox_rect.collidepoint(pos):
            checkbox_state = not checkbox_state

        colors = {}
        for key, rect in input_boxes.items():
            active = rect.collidepoint(pos)
            colors[key] = pygame.Color('dodgerblue2') if active else pygame.Color('lightskyblue3')

        return checkbox_state, input_boxes, colors


    def _handle_formation_click(self, pos, inputs):
        for formation, rect in self.formation_rects:
            if rect.collidepoint(pos):
                inputs["def_default_formation"] = formation
                return  # Select only one at a time

        for formation, rect in self.moving_formation_rects:
            if rect.collidepoint(pos):
                inputs["def_moving_formation"] = formation
                return


    def _validate_inputs(self, inputs, custom):
        if not inputs["total_num_def"].isdigit():
            return False, "Enter a valid number for total defender ships."

        if custom and not all(inputs[key].isdigit() for key in ["num_sonar_def", "num_heli_def", "num_decoy_def", "num_default_def"]):
            return False, "All custom defender values must be valid numbers."

        if custom and sum(int(inputs[key]) for key in ["num_sonar_def", "num_heli_def", "num_decoy_def", "num_default_def"]) != int(inputs["total_num_def"]):
            return False, "Sum of custom types must equal total defender ships."

        try:
            lat, lon = map(float, inputs["map_center"].split(','))
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return False, "Map coordinates must be valid lat,long."
        except Exception:
            return False, "Map coordinates must be in lat,long format."

        try:
            base_lat, base_lon = map(float, inputs["base_location"].split(','))
            if not (-90 <= base_lat <= 90 and -180 <= base_lon <= 180):
                return False, "Base location coordinates must be valid lat,long."
        except Exception:
            return False, "Base location must be in lat,long format."


        if not inputs["zoom"].isdigit():
            return False, "Zoom level must be a valid number."

        if not (1 <= int(inputs["zoom"]) <= 12):
            return False, "Zoom level must be between 1 and 12."

        return True, ""



    def _handle_key_input(self, event, inputs, boxes, colors):
        for key, rect in boxes.items():
            if colors[key] == pygame.Color('dodgerblue2'):  # Active box
                if event.key == pygame.K_RETURN:

                    colors[key] = pygame.Color('lightskyblue3')  # Deactivate on Enter
                elif event.key == pygame.K_BACKSPACE:
                    inputs[key] = inputs[key][:-1]
                elif event.unicode.isdigit() or event.unicode in ['.', ',']:
                    inputs[key] += event.unicode
        return inputs, colors



    def _finalize_user_inputs(self, inputs, custom):
        num_defenders = int(inputs["total_num_def"])
        num_sonar = int(inputs["num_sonar_def"]) if custom else 0
        num_heli = int(inputs["num_heli_def"]) if custom else 0
        num_decoy = int(inputs["num_decoy_def"]) if custom else 0
        num_default = int(inputs["num_default_def"]) if custom else 0
        lat, lon = map(float, inputs["map_center"].split(','))
        zoom = int(inputs["zoom"])
        formation = inputs["def_default_formation"]
        moving_formation = inputs["def_moving_formation"]
        base_lat, base_lon = map(float, inputs["base_location"].split(','))
        #pygame.display.quit()
        #pygame.display.init()
        return num_defenders, custom, num_sonar, num_heli, num_decoy, num_default, formation, [lat, lon], zoom, [base_lat, base_lon], moving_formation

class Renderer:
    def __init__(self, env, map_generator):
        """Renderer for displaying real-world maps with accurate latitude/longitude grid lines."""
        self.env = env
        self.map_generator = map_generator

    def _get_scaled_size(self, base_size):
        """Calculate entity size based on zoom level with smooth scaling"""
        min_scale = 0.5  # Minimum scale to maintain visibility
        max_scale = 2.0  # Maximum scale to prevent objects from being too large
        zoom_ratio = self.env.zoom / 6.0  # Using zoom level 6 as baseline

        # Apply smooth exponential scaling
        if self.env.zoom < 6:
            # Gradual reduction when zooming out
            scale = max(min_scale, (zoom_ratio ** 1.5))
        else:
            # Gradual increase when zooming in
            scale = min(max_scale, zoom_ratio)

        return int(base_size * scale)

    def _render_ship(self, ship, color=(255, 255, 255), base_rect_size=20, line_thickness=1):
        """
        Render a ship as a rectangle and draw its firing range as a circle.
        Also displays the latitude and longitude of the ship below it.
        """
        if ship.ship_health > 0:
            ship.update_pixel_position()
            x, y = ship.x, ship.y

            # Scale sizes based on zoom level
            rect_size = self._get_scaled_size(base_rect_size)

            # Calculate firing range scale based on zoom level
            # Use exponential scaling to make ranges get dramatically smaller as we zoom out
            zoom_factor = (self.env.zoom / 6.0) ** 2  # Exponential reduction based on zoom

            # Additional reduction factor for zooming out
            if self.env.zoom < 6:
                # More aggressive reduction when zoomed out
                zoom_out_factor = (self.env.zoom / 6.0) ** 1.5  # Extra reduction for zoom out
                range_scale = zoom_factor * zoom_out_factor
            else:
                # Normal scaling when zoomed in
                range_scale = min(2.0, zoom_factor)  # Cap the maximum scale at 2x

            # Ensure minimum visibility
            scaled_range = int(ship.firing_range * range_scale)
            if scaled_range > 0:
                scaled_range = max(rect_size * 2, min(scaled_range, ship.firing_range))  # Between 2x ship size and original range

            # Draw ship (rectangle) centered on its position
            pygame.draw.rect(self.env.screen, color,
                           (int(x) - rect_size // 2, int(y) - rect_size // 2,
                            rect_size, rect_size))

            # Draw firing range (circle) around the ship
            if ship.firing_range > 0:
                pygame.draw.circle(self.env.screen, color, (int(x), int(y)),
                                scaled_range, line_thickness)

            # Render lat/lon text below the ship
            self.map_generator._render_latlon(x, y, rect_size)

    def _render_torpedoes(self, ship, color=(255, 255, 255)):
        """
        Render all active torpedoes fired by a ship.
        """
        torpedo_size = self._get_scaled_size(5)  # Base torpedo size of 5
        for torpedo in ship.torpedoes:
            if not torpedo.target_hit:
                pygame.draw.circle(self.env.screen, color,
                                (int(torpedo.x), int(torpedo.y)),
                                torpedo_size)

    def _render_decoyM(self, ship, color=(255, 165, 0)):
        missile_size = self._get_scaled_size(5)  # Base missile size of 5
        for missile in ship.decoy_missile:
            if not missile['target_hit']:
                if 'x' not in missile or 'y' not in missile:
                    missile['x'], missile['y'] = self.map_generator._latlon_to_pixels(missile['lat'], missile['lon'])

                pygame.draw.circle(self.env.screen, color,
                                (int(missile['x']), int(missile['y'])),
                                missile_size)

    def draw_defender_top_icon(self, defender):
        ship_type = defender.ship_type
        x, y = int(defender.x), int(defender.y)

        if defender.ship_health > 0:
            base_size = 5
            size = self._get_scaled_size(base_size)

            if ship_type == "def_heli":
                # Yellow triangle (upward)
                points = [(x, y - size),
                         (x - size, y + size),
                         (x + size, y + size)]
                pygame.draw.polygon(self.env.screen, (255, 255, 0), points)

            elif ship_type == "def_sonar":
                # White circle
                pygame.draw.circle(self.env.screen, (255, 255, 255), (x, y), size)

            elif ship_type == "def_decoyM":
                rect_size = self._get_scaled_size(12)
                pygame.draw.rect(self.env.screen, (255, 140,  0),
                               (int(x) - rect_size // 2,
                                int(y) - rect_size // 2,
                                rect_size, rect_size))

    def _render_helicopter(self, latlon):
        """
        Render the helicopter and its firing range with consistent geographic scaling.
        """
        if self.env.helicopManager.helicopter_lat is not None:
            # Use the helicopter's stored pixel coordinates
            x, y = self.env.helicopManager.helicopter_x, self.env.helicopManager.helicopter_y

            # Calculate helicopter size with proper zoom scaling
            base_size = 15  # Base size at zoom level 6
            min_size = 10   # Absolute minimum size for visibility
            max_size = 25   # Maximum size to prevent being too large

            # Use same scaling logic as ships for consistency
            zoom_factor = (self.env.zoom / 6.0) ** 2
            if self.env.zoom < 6:
                zoom_out_factor = (self.env.zoom / 6.0) ** 1.5
                size_scale = zoom_factor * zoom_out_factor
            else:
                size_scale = min(2.0, zoom_factor)

            size = int(max(min_size, min(max_size, base_size * size_scale)))

            # Calculate firing range circle radius in screen coordinates
            range_km = self.env.helicop_range
            lat_deg = range_km / 111.0  # Convert km to degrees latitude

            # Calculate range using the same method as ships for consistency
            range_scale = size_scale  # Use same scaling as entity size
            scaled_range = int(self.env.helicop_range * range_scale)

            # Ensure range remains visible but doesn't exceed realistic proportions
            min_range = size * 2
            max_range = self.env.helicop_range  # Original maximum range
            scaled_range = max(min_range, min(scaled_range, max_range))

            # Draw perfectly straight helicopter triangle
            # Define points for an upward-pointing isosceles triangle
            points = [
                (x, y - size),           # Top point
                (x - size, y + size),    # Bottom left
                (x + size, y + size)     # Bottom right
            ]

            # Draw helicopter shape with straight lines
            pygame.draw.polygon(self.env.screen, (255, 255, 0), points)

            # Draw detection range circle
            pygame.draw.circle(self.env.screen, (255, 255, 0), (x, y), scaled_range, 1)

            # Draw position label
            self.map_generator._render_latlon(x, y, size * 2)

    def _render_attacker(self):
        """Render the attacker ship and its torpedoes."""
        self._render_ship(self.env.attacker_ship, color=(255, 0, 0))
        self._render_torpedoes(self.env.attacker_ship, color=(255, 0, 0))

    def _render_defenders(self):
        """Render all defender ships and their associated elements."""
        for defender in self.env.defender_ships:
            # Render the main defender ship
            self._render_ship(defender, color=(0, 0, 255))

            # Render torpedoes for each defender
            self._render_torpedoes(defender, color=(0, 0, 255))

            # Render decoy missiles if the defender has them
            if defender.ship_type == 'def_decoyM':
                self._render_decoyM(defender)

            # Draw the top icon for each defender type
            self.draw_defender_top_icon(defender)



    def _display_health(self):
        """
        Display the health of the attacker and the HVU at the bottom-left corner of the screen.
        """
        # Get screen dimensions
        screen_width, screen_height = self.env.width, self.env.height

        # Render health information with black text
        attacker_health_text = self.env.font.render(f"Attacker Health: {self.env.attacker_ship.ship_health}", True, (255, 0, 0))
        hvu_health_text = self.env.font.render(f"HVU Health: {self.env.hvu.ship_health}", True, (0, 100, 0))

        # Get text dimensions
        text_height = attacker_health_text.get_height()

        # Position at bottom-left corner
        bottom_offset = 10  # Space from the bottom
        left_offset = 10  # Space from the left

        self.env.screen.blit(attacker_health_text, (left_offset, screen_height - 2 * text_height - bottom_offset))
        self.env.screen.blit(hvu_health_text, (left_offset, screen_height - text_height - bottom_offset))


    def _render_base(self):
        """
        Render the base location as a green big circle with a black border.
        """
        if hasattr(self.env, "base_location"):

            # First, draw the black border (slightly bigger radius)
            pygame.draw.circle(self.env.screen, (0, 0, 0), self.env.base_location_inPixels, 32)  # Black border with radius 32

            # Then, draw the actual base circle on top
            pygame.draw.circle(self.env.screen, (0, 200, 0), self.env.base_location_inPixels, 30)  # Green circle with radius 30

class NavalShipEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
        self,

        screen_width: int = 1000,
        screen_height: int = 600,
        env_name: str = "Naval Ship Environment",

        # Default Configuration of defenders and its formation
        tota_num_def: int = 5,
        num_def_with_sonar: int = 1,
        num_def_with_helicopter: int = 2,
        num_def_with_decoy: int = 1,
        num_default_def: int = 1,
        def_default_formation: str = "semicircle",
        map_center: List [float] = [3.0000, 86.0000],
        zoom: int = 6,
        base_location: List [float] = [3.8, 87.5000],

        # --- Ship Configuration ---
        hvu_ship: dict = None,
        att_ship: dict = None,
        def_ship: dict = None,
        def_sonar: dict = None,
        def_heli: dict = None,
        def_decoyM: dict = None,

        # --- Formation and Movement ---
        def_moving_formation: str = "wedge",

        # --- Helicopter Configuration ---
        helicop_path_radius: int = 200,
        helicop_range: int = 150,
        helicop_speed: float = 2.0
    ):

        super(NavalShipEnv, self).__init__()

        # Initialize Pygame and basic display
        pygame.init()
        self.width, self.height = screen_width, screen_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(env_name)
        self.clock = pygame.time.Clock()
        # Zoom buttons
        self.zoom_in_button = pygame.Rect(self.width - 60, 20, 30, 30)
        self.zoom_out_button = pygame.Rect(self.width - 25, 20, 30, 30)


        self.font = pygame.font.SysFont('Arial', 15)

        # --- Store Formations and Helicopter Config ---
        self.avail_def_default_formations = ["semicircle", "circle", "triangle"]
        self.avail_def_moving_formations = ["line", "wedge", "semicircle", "circle", "triangle"]
        self.def_moving_formation = def_moving_formation

        self.helicop_path_radius = helicop_path_radius
        self.helicop_range = helicop_range
        self.helicop_speed = helicop_speed


        # Store Default Configuration of defenders and its formation
        self.num_defenders = tota_num_def
        self.num_def_with_sonar = num_def_with_sonar
        self.num_def_with_helicopter = num_def_with_helicopter
        self.num_def_with_decoy = num_def_with_decoy
        self.num_default_def = num_default_def
        self.def_default_formation = def_default_formation
        self.map_center = map_center
        self.base_location_inPixels = None
        self.zoom = zoom
        self.base_location = base_location
        self.def_moving_formation = def_moving_formation

        # Get User Input for environemnt configuration
        (
            self.num_defenders,
            self.custmise_def,
            self.num_def_with_sonar,
            self.num_def_with_helicopter,
            self.num_def_with_decoy,
            self.num_default_def,
            self.def_formation,
            self.map_center,
            self.zoom,
            self.base_location,
            self.def_moving_formation
        ) = UIManager(self).get_user_input_screen()
                # 🧹 Clear leftover UI screen after input
        self.screen.fill((0, 0, 0))
        pygame.display.flip()



        # Define Observation Space (Attacker + HVU + all Defenders)
        total_ships = self.num_defenders + 2
        self.observation_space = spaces.Box(
            low=np.zeros(2 * total_ships, dtype=np.float32),
            high=np.array([self.width, self.height] * total_ships, dtype=np.float32),
            dtype=np.float32
        )

        # Define Attacker's Action Space: [Stay, Up, Down, Left, Right]
        self.action_space = spaces.Discrete(5)



        # --- Default Ship Templates ---
        self.ship_templates = {
            "hvu_ship": copy.deepcopy(hvu_ship) if hvu_ship else {
                'lat': 3.0,
                'lon': 86.0,
                'speed': 15,
                'ship_type': 'HVU',
                'firing_range': 0,
                'ship_health': 10,
                'torpedo_count': 0,
            },
            "att_ship": copy.deepcopy(att_ship) if att_ship else {
                'lat': 2.8,
                'lon': 85.6,
                'speed': 30,
                'ship_type': 'attacker_ship',
                'firing_range': 200,
                'ship_health': 10,
                'reload_delay': 0.5,
                'target_delay': 0.2,
                'torpedo_count': 100,
                'torpedo_fire_speed': 2.0,
                'torpedo_damage': 1,
            },
            "def_ship": copy.deepcopy(def_ship) if def_ship else {
                'lat': 3.05,
                'lon': 86.05,
                'speed': 15,
                'ship_type': 'defender',
                'firing_range': 100,
                'reload_delay': 0.5,
                'target_delay': 0.2,
                'helicop_count': 0,
                'torpedo_count': 100,
                'torpedo_fire_speed': 2.0,
                'torpedo_damage': 1,
                'decoyM_count': 0
            },
            "def_sonar": copy.deepcopy(def_sonar) if def_sonar else {
                'lat': 3.05,
                'lon': 85.95,
                'speed': 1.5,
                'ship_type': 'def_sonar',
                'firing_range': 150,
                'reload_delay': 0.5,
                'target_delay': 0.2,
                'helicop_count': 0,
                'torpedo_count': 100,
                'torpedo_fire_speed': 2.0,
                'torpedo_damage': 1,
                'decoyM_count': 0
            },
            "def_heli": copy.deepcopy(def_heli) if def_heli else {
                'lat': 2.95,
                'lon': 86.05,
                'speed': 1.5,
                'ship_type': 'def_heli',
                'firing_range': 100,
                'reload_delay': 0.5,
                'target_delay': 0.2,
                'helicop_count': 1,
                'torpedo_count': 100,
                'torpedo_fire_speed': 2.0,
                'torpedo_damage': 1,
                'decoyM_count': 0
            },
            "def_decoyM": copy.deepcopy(def_decoyM) if def_decoyM else {
                'lat': 2.95,
                'lon': 85.95,
                'speed': 1.5,
                'ship_type': 'def_decoyM',
                'firing_range': 100,
                'reload_delay': 0.5,
                'target_delay': 0.2,
                'helicop_count': 0,
                'torpedo_count': 100,
                'torpedo_fire_speed': 2.0,
                'torpedo_damage': 1,
                'decoyM_count': 100,
                'decoyM_speed': 3.0,
                'decoyM_blast_range': 2.0,
            }
        }

        # Call environment reset
        self.reset()


    def reset(self):
        # Reset environment state
        self.info = {
            'collision': 0,
            'attacker destroyed': 0,
            'HVU destroyed': 0,
            'step count': 0,
            'Returned to Base': 0
        }
        self.reward = 0
        self.done = False
        self.paused = False
        self.attacker_fired = False


        # Initialize HVU (High Value Unit)
        self.hvu = Ship(
            self,
            ship_id=0,
            screen_width=self.width,
            screen_height=self.height,
            **self.ship_templates["hvu_ship"]
        )

        # Initialize Attacker Ship
        self.attacker_ship = Ship(
            self,
            ship_id=1,
            screen_width=self.width,
            screen_height=self.height,
            **self.ship_templates["att_ship"]
        )

        self.mapGenerator = MapGenerator(self, map_center=self.map_center, zoom=self.zoom)
        # Initialize Defender Ships
        self.defender_ships = self.define_defenders()

        # Combine all ships
        self.ships = [self.attacker_ship, self.hvu] + self.defender_ships

        # Initialize Map Renderer
        #self.mapGenerator = MapGenerator(self, map_center=self.map_center, zoom=self.zoom)
        self.background_image = pygame.image.load(self.mapGenerator.map_image_path)
        self.background_image = pygame.transform.scale(self.background_image, (self.width, self.height))

        self.renderer = Renderer(self, self.mapGenerator)

        # Initialize Managers
        self.defence_system = DefenseSystem(self)
        self.firemechanism = FireMechanism(self)
        self.helicopManager = HelicopterManager(self)
        self.DecoyMissileManager = DecoyMissileManager(self)

        #Convert Base Location from lat, long to pixels for calculation
        self.base_location_inPixels = self.mapGenerator._latlon_to_pixels(self.base_location[0], self.base_location[1])


        # Return initial observation
        return self._get_obs()

    def _reload_map_image(self):
        self.mapGenerator = MapGenerator(self, map_center=self.map_center, zoom=self.zoom)
        self.background_image = pygame.image.load(self.mapGenerator.map_image_path)
        self.background_image = pygame.transform.scale(self.background_image, (self.width, self.height))

        self.base_location_inPixels = self.mapGenerator._latlon_to_pixels(
            self.base_location[0], self.base_location[1]
        )
        for ship in self.ships:
            ship.update_pixel_position()

# ✅ Update helicopter if it stores lat/lon
        if self.helicopManager.helicopter_active and hasattr(self.helicopManager, "helicopter_lat"):
            self.helicopManager.helicopter_position = np.array(
            self.mapGenerator._latlon_to_pixels(
                self.helicopManager.helicopter_lat,
                self.helicopManager.helicopter_lon
            ), dtype=np.float64
        )



    def define_defenders(self):
        defender_ships = []
        current_id = 2  # Start from 2 (0: HVU, 1: Attacker)

        def add_ship_from_template(template_name):
            nonlocal current_id
            # Get a copy of the ship template
            template = copy.deepcopy(self.ship_templates[template_name])

            # Create the Ship using unpacked template values + assigned ship_id
            ship = Ship(
                self,
                ship_id=current_id,
                screen_width=self.width,
                screen_height=self.height,
                **template
            )
            defender_ships.append(ship)
            current_id += 1

        if not self.custmise_def:
            # Default setup with 5 mixed defenders
            add_ship_from_template("def_heli")
            add_ship_from_template("def_decoyM")
            add_ship_from_template("def_sonar")
            add_ship_from_template("def_decoyM")
            add_ship_from_template("def_heli")

            # Fill remaining slots with basic defenders
            for _ in range(self.num_defenders - 5):
                add_ship_from_template("def_ship")

        else:
            # User custom configuration
            for _ in range(self.num_def_with_sonar):
                add_ship_from_template("def_sonar")

            for _ in range(self.num_def_with_helicopter):
                add_ship_from_template("def_heli")

            for _ in range(self.num_def_with_decoy):
                add_ship_from_template("def_decoyM")

            # Fill remaining defenders with default if user count is short
            total_custom = self.num_def_with_sonar + self.num_def_with_helicopter + self.num_def_with_decoy
            for _ in range(self.num_defenders - total_custom):
                add_ship_from_template("def_ship")

        return defender_ships


    def _get_obs(self):
        # Return the positions and headings of all ships
        observation = []
        for ship in self.ships:
            observation.extend(ship.get_position())
        return np.array(observation, dtype=np.float32)


    def step(self, action):
        # 1. Move ships (attacker, defenders, HVU)
        self._movements(action)

        # 1.5 Apply repulsion to maintain spacing between defenders
        self.defence_system.apply_repulsion()

        # 2. Update torpedoes and decoys
        self.firemechanism._update_torpedo_position()
        self.DecoyMissileManager.update_decoy_missile()

        # 3. Calculate reward and handle torpedo firing logic
        self.reward += self._calculate_reward()

        # ✅ 4. Now check for ship collisions AFTER firing happens
        if self.firemechanism.check_collisions():
            self.reward -= 100  # Heavy penalty for collision
            self.info['collision'] += 1
            self.done = True
            return self._get_obs(), self.reward, self.done, self.info

        # 5. Increment step counter
        self.info['step count'] += 1
        return self._get_obs(), self.reward, self.done, self.info



    def _movements(self, action):
        self._move_attacker(action)
        #self.firemechanism._update_torpedo_position()
        self.DecoyMissileManager.update_decoy_missile()

        if not self.attacker_ship.target_in_range(self.hvu) and self.check_los_attacker():
            self.attacker_ship.move_ship_to_coordinates((self.hvu.lat, self.hvu.lon))

        self.move_defenders()

        if self.attacker_fired:
            self.move_hvu_to_base()
            self._check_hvu_reached_base()



    def _check_hvu_reached_base(self):
        """
        Check if the HVU has reached the base using haversine distance.
        """
        distance = haversine_distance(self.hvu.lat, self.hvu.lon, 
                                    self.base_location[0], self.base_location[1])
        
        if distance < 0.5:  # Within 0.5 km of base
            self.done = True  # End the episode
            self.info['Returned to Base'] += 1

    def move_hvu_to_base(self):
        """
        Move the HVU towards the base location using geographic (lat/lon) displacement.
        """
        current_lat, current_lon = self.hvu.lat, self.hvu.lon
        base_lat, base_lon = self.base_location

    # Calculate haversine distance and bearing
        distance_km = haversine_distance(current_lat, current_lon, base_lat, base_lon)
        if distance_km < 0.1:  # Already close
            return

    # Compute bearing
        bearing_rad = math.atan2(
            math.radians(base_lon - current_lon),
            math.radians(base_lat - current_lat)
        )

    # Convert speed (km per step) into degrees displacement
        delta_lat = self.hvu.speed * math.cos(bearing_rad) / 111
        delta_lon = self.hvu.speed * math.sin(bearing_rad) / (111 * math.cos(math.radians(current_lat)))

    # Update HVU's lat/lon
        self.hvu.lat += delta_lat
        self.hvu.lon += delta_lon
        self.hvu.update_pixel_position()



    def move_defenders(self):
        # Handle defense system movements if active


        if self.defence_system.defense_active:
            self.defence_system.handle_defense_mechanism(formation_type=self.def_moving_formation) # tri, line, wedge, circle

        else:
            # keep defenders in current formation
            self.defence_system.move_defenders_in_formation(self.def_formation)

        self.defence_system.check_for_defense_activation()

        # Movement of Defender's Helicopter
        self.helicopManager.move_defenders_helicop()


    def _move_attacker(self, action):
        """Move the attacker ship based on the provided action index."""

        # Mapping action index to movement heading in degrees
        action_heading_map = {
            0: 90,    # Up
            1: 270,   # Down
            2: 180,   # Left
            3: 0,     # Right
            4: 45,    # Up-Right
            5: 135,   # Up-Left
            6: 225,   # Down-Left
            7: 315    # Down-Right
            # 8: No movement
        }

        heading = action_heading_map.get(action)

        if heading is not None:
            self.attacker_ship.move_ship_to_direction(heading=heading)
        elif action == 8:
            pass  # No movement
        else:
            print(f"[WARNING] Invalid action received: {action}")

#         return


    def _calculate_reward(self):
        reward = 0  # Step penalty

        # Calculate distance between attacker and HVU using haversine
        distance_to_hvu = haversine_distance(
            self.attacker_ship.lat, self.attacker_ship.lon,
            self.hvu.lat, self.hvu.lon
        )

        hvu_in_att_range = self.attacker_ship.target_in_range(self.hvu)

        if hvu_in_att_range:
            reward += 2

        # Negative reward for moving too far from HVU (outside attacker's firing range)
        if not hvu_in_att_range:
            reward -= (distance_to_hvu / 10)

        # Check if the attacker is within any defender's firing range
        in_range_defenders = self.defence_system.attacker_within_defender_range()

        if in_range_defenders:
            reward -= 5

        # Reward for taking HVU in firing range and being out of defender's range
        if hvu_in_att_range and not in_range_defenders:
            reward += 20

        reward += self.firemechanism._handle_firing_mechanics()

        self.DecoyMissileManager.handle_DecoyM_firing_mechanics()

        return reward



    def check_los_attacker(self):
        """
        Check if any defender ship is blocking the line of sight between the attacker and the HVU ship.
        Returns False if a defender is in the way, otherwise returns True.
        """
        attacker_pos = self.attacker_ship.get_position()
        hvu_pos = self.hvu.get_position()

        # Loop through each defender and check if they are blocking the line of sight
        for defender in self.defender_ships:
            defender_pos = defender.get_position()
            if self.check_if_blocking_los(attacker_pos, hvu_pos, defender_pos):
                return False  # Defender is blocking the line of sight

        return True  # No defender is blocking the line of sight


    def check_los_defender(self, defender):
        """
        Check if the defender has a clear line of sight to fire at the attacker.
        Ensures no other defender or the HVU is blocking the LOS.
        """
        defender_pos = defender.get_position()
        attacker_pos = self.attacker_ship.get_position()

    # Check if HVU is blocking the line
        if self.check_if_blocking_los(defender_pos, attacker_pos, self.hvu.get_position()):
            return False

    # Check if any other defender is blocking the LOS
        for other in self.defender_ships:
            if other.ship_id == defender.ship_id:
                continue  # Skip self
            if self.check_if_blocking_los(defender_pos, attacker_pos, other.get_position()):
                return False

        return True  # Clear line of sight



    def check_line_intersection(self, A, B, C, D):
        """
        Helper function to check if two line segments (AB and CD) intersect.
        """
    # Convert all to NumPy arrays (fix for subtraction error)
        A = np.array(A)
        B = np.array(B)
        C = np.array(C)
        D = np.array(D)

        def cross_product(v1, v2):
            return v1[0] * v2[1] - v1[1] * v2[0]

        AB = B - A
        AC = C - A
        AD = D - A
        CD = D - C
        CA = A - C
        CB = B - C

        cross1 = cross_product(AB, AC)
        cross2 = cross_product(AB, AD)
        cross3 = cross_product(CD, CA)
        cross4 = cross_product(CD, CB)

        if (cross1 * cross2 < 0) and (cross3 * cross4 < 0):
            return True

        return False



    def check_if_blocking_los(self, start_pos, end_pos, blocker_center, blocker_size=(20, 20)):
        """
        Checks if a rectangular blocker ship is in the line of sight (LoS) between start and end positions.
        Uses line-segment intersection instead of just angle thresholding.

        Args:
        - start_pos (np.array): Start position (x, y).
        - end_pos (np.array): End position (x, y).
        - blocker_center (np.array): Center of the blocking ship (x, y).
        - blocker_size (tuple): (width, height) of the blocker ship.

        Returns:
        - bool: True if blocking, False otherwise.
        """
        # Compute LoS vector
        line_vector = np.array(end_pos) - np.array(start_pos)

        if np.linalg.norm(line_vector) == 0:
            return False  # Avoid division by zero

        # Define blocker boundaries
        half_width, half_height = blocker_size[0] / 2, blocker_size[1] / 2
        blocker_corners = [
            blocker_center + np.array([-half_width, -half_height]),  # Bottom-left
            blocker_center + np.array([half_width, -half_height]),   # Bottom-right
            blocker_center + np.array([-half_width, half_height]),   # Top-left
            blocker_center + np.array([half_width, half_height])     # Top-right
        ]

        # Define the four edges of the rectangle as line segments
        blocker_edges = [
            (blocker_corners[0], blocker_corners[1]),  # Bottom edge
            (blocker_corners[1], blocker_corners[3]),  # Right edge
            (blocker_corners[3], blocker_corners[2]),  # Top edge
            (blocker_corners[2], blocker_corners[0])   # Left edge
        ]

        # Check if the line of sight intersects any of the blocker edges
        for edge in blocker_edges:
            if self.check_line_intersection(start_pos, end_pos, edge[0], edge[1]):
                return True  # If LoS intersects any edge, it's blocked

        return False  # No intersection → No blockage


    def check_for_collisions_while_ship_moves(self, ship, all_ships, ship_target_pos):
        ship_current_pos = ship.get_position()
        for other_ship in all_ships:
            if ship != other_ship:
                other_ship_pos = other_ship.get_position()
                if ship.target_in_range(other_ship_pos):
                    # Check if ship j is in the way of the line from current_pos_i to target_pos
                    if self.check_if_blocking_los(ship_current_pos, ship_target_pos, other_ship_pos):
                        return other_ship, True  # Collision detected

        return None, False  # No collision detected


    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.background_image, (0, 0))  # Background map

    # Draw ships and simulation elements
        self.renderer._render_ship(self.hvu, color=(0, 255, 0))
        self.renderer._render_attacker()
        self.renderer._render_defenders()

        # Pass lat/lon position for helicopter if active
        if self.helicopManager.helicopter_active:
            helicopter_pos = (self.helicopManager.helicopter_lat, self.helicopManager.helicopter_lon)
            self.renderer._render_helicopter(helicopter_pos)

        self.renderer._render_base()
        self.renderer._display_health()

    # Draw zoom buttons
        # --- Zoom In and Zoom Out Buttons ---
        pygame.draw.rect(self.screen, (0, 200, 0), self.zoom_in_button)    # Green + button
        pygame.draw.rect(self.screen, (200, 0, 0), self.zoom_out_button)   # Red - button

        font = pygame.font.SysFont('Arial', 24, bold=True)
        plus_text = font.render("+", True, (255, 255, 255))
        minus_text = font.render("-", True, (255, 255, 255))
        self.screen.blit(plus_text, (self.zoom_in_button.x + 7, self.zoom_in_button.y + 2))
        self.screen.blit(minus_text, (self.zoom_out_button.x + 7, self.zoom_out_button.y + 2))

        pygame.display.update()
        self.clock.tick(60)


    def _update_map_zoom(self):
        self.mapGenerator.zoom = self.zoom
        self.mapGenerator.map_initialized = False  # Force reinitialization
        self.mapGenerator.map_initialise()
        self.background_image = pygame.image.load(self.mapGenerator.map_image_path)
        self.background_image = pygame.transform.scale(self.background_image, (self.width, self.height))


    def close(self):
        """Closes the Pygame window."""
        pygame.quit()


    def seed(self, seed=None):
        """Sets the random seed."""
        random.seed(seed)
        np.random.seed(seed)

"""# Define the Environment Details"""

hvu_ship = {
    'lat': 5.0,
    'lon': 85.0,
    'speed': 1.5,
    'ship_type': 'HVU',
    'firing_range': 0,
    'ship_health': 10,
    'torpedo_count': 0,
}

att_ship = {
    'lat': 3.5,   # moved further south-west from HVU
    'lon': 83.0,
    'speed': 1.5,
    'ship_type': 'attacker_ship',
    'firing_range': 200,
    'ship_health': 10,
    'reload_delay': 0.5,
    'target_delay': 0.2,
    'torpedo_count': 100,
    'torpedo_fire_speed': 2.0,
    'torpedo_damage': 1,
}

def_ship = {
    'lat': 5.3,
    'lon': 85.3,
    'speed': 1.5,
    'ship_type': 'defender',
    'firing_range': 100,
    'reload_delay': 0.5,
    'target_delay': 0.2,
    'helicop_count': 0,
    'torpedo_count': 100,
    'torpedo_fire_speed': 2.0,
    'torpedo_damage': 1,
    'decoyM_count': 0
}

def_sonar = {
    'lat': 5.3,
    'lon': 84.7,
    'speed': 1.5,
    'ship_type': 'def_sonar',
    'firing_range': 150,
    'reload_delay': 0.5,
    'target_delay': 0.2,
    'helicop_count': 0,
    'torpedo_count': 100,
    'torpedo_fire_speed': 2.0,
    'torpedo_damage': 1,
    'decoyM_count': 0
}

def_heli = {
    'lat': 4.7,
    'lon': 85.3,
    'speed': 3,
    'ship_type': 'def_heli',
    'firing_range': 100,
    'reload_delay': 0.5,
    'target_delay': 0.2,
    'helicop_count': 1,
    'torpedo_count': 100,
    'torpedo_fire_speed': 2.0,
    'torpedo_damage': 1,
    'decoyM_count': 0
}

def_decoyM = {
    'lat': 4.7,
    'lon': 84.7,
    'speed': 1.5,
    'ship_type': 'def_decoyM',
    'firing_range': 100,
    'reload_delay': 0.5,
    'target_delay': 0.2,
    'helicop_count': 0,
    'torpedo_count': 100,
    'torpedo_fire_speed': 2.0,
    'torpedo_damage': 1,
    'decoyM_count': 100,
    'decoyM_speed': 3.0,
    'decoyM_blast_range': 2.0
}

"""# Game Play"""

if __name__ == "__main__":

    # Initialise the Environment
    env = NavalShipEnv(
        screen_width=1000,
        screen_height=600,
        env_name="Naval Ship Environment",

        tota_num_def=6,
        num_def_with_sonar=1,
        num_def_with_helicopter=2,
        num_def_with_decoy=2,
        num_default_def=1,
        def_default_formation="semicircle",
        map_center=[3.0000, 86.0000],
        zoom=6,
        base_location=[6.0,96.5],

        hvu_ship=hvu_ship,
        att_ship=att_ship,
        def_ship=def_ship,
        def_heli=def_heli,
        def_decoyM=def_decoyM,
        def_sonar=def_sonar,

        def_moving_formation="line",
        helicop_path_radius=200,
        helicop_range=150,
        helicop_speed=2.0
    )

    obs = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    env.paused = not env.paused
                elif event.key == pygame.K_q:
                    running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if env.zoom_in_button.collidepoint(event.pos) and env.zoom < 12:
                    env.zoom += 1
                    env._reload_map_image()

                elif env.zoom_out_button.collidepoint(event.pos) and env.zoom > 1:
                    env.zoom -= 1
                    env._reload_map_image()

        if not env.paused and not env.done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            env.render()

    env.close()



