
# Imports


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
import random
from dataclasses import dataclass, field
from collections import deque
from typing import Callable, Deque, List, Optional
import heapq
import json

from tkinter import Tk
from tkinter.filedialog import askopenfilename

"""#Data Extraction and Histogram-Based Analysis of Tube Ride Service Times"""


# Select Excel file
Tk().withdraw()  # hide tkinter window
excel_file = askopenfilename(
    title="Select Excel file",
    filetypes=[("Excel files", "*.xlsx")]
)

column_name = "time_minutes"

def read_column(sheet_name):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    return df[column_name].dropna()


# sheets
name_map = {
    "מגלשת אבובים גדולה": "Big Tube Slide",
    "מגלשת אבובים קטנה ": "Small Tube Slide"
}
sheets = ["מגלשת אבובים גדולה", "מגלשת אבובים קטנה "]

for sheet in sheets:
    data = read_column(sheet)
    english_name = name_map[sheet]

    print(f"\n--- {sheet} ---")
    print("Count:", len(data))
    print("Mean:",  np.mean(data))
    print("Median:", np.median(data))
    print("Std:",   np.std(data, ddof=1))
    print("Min:",   np.min(data))
    print("Max:",   np.max(data))

    plt.hist(data, bins=20, edgecolor="black")
    plt.title(f"Histogram – {english_name}")
    plt.xlabel("time_minutes")
    plt.ylabel("frequency")
    plt.show()

"""#Distribution Fitting and Goodness-of-Fit Analysis"""

# ---------------------------------------------------------
# 1. MLE function
# ---------------------------------------------------------

def calculate_mle_normal(data):
    n = len(data)
    mu_hat = np.sum(data) / n
    sigma_hat = np.sqrt(np.sum((data - mu_hat)**2) / n)
    return mu_hat, sigma_hat

def calculate_mle_expon(data):
    n = len(data)
    mean_val = np.sum(data) / n
    lambda_hat = 1 / mean_val
    scale_hat = mean_val
    return lambda_hat, scale_hat

# ---------------------------------------------------------
# 2. KS TEST
# ---------------------------------------------------------

def manual_ks_test(data, dist_name, params, alpha=0.05):

    #  A. Sort the data
    data_sorted = np.sort(data)
    n = len(data)

    # B. Build the ECDF step indices (1..n)
    i = np.arange(1, n + 1)

    #  Compute the theoretical CDF values at the sorted sample points
    if dist_name == 'norm':
        mu, sigma = params
        F = stats.norm.cdf(data_sorted, loc=mu, scale=sigma)
    elif dist_name == 'expon':
        scale = params[1]
        # Manual exponential CDF: F(x) = 1 - exp(-x/scale)
        F = 1 - np.exp(-data_sorted / scale)

    # D. Compute the  KS statistic D (max vertical distance)
    diff_plus = np.max(i / n - F)
    diff_minus = np.max(F - (i - 1) / n)
    D_raw = max(diff_plus, diff_minus)

    # E. Apply Stephens' correction + select the fixed critical value (alpha = 0.05)

    if dist_name == 'norm':
        D_adjusted = D_raw * (np.sqrt(n) - 0.01 + (0.85 / np.sqrt(n)))
        critical_value = 0.895

    elif dist_name == 'expon':
        D_adjusted = (D_raw - (0.2 / n)) * (np.sqrt(n) + 0.26 + (0.5 / np.sqrt(n)))
        critical_value = 1.094

    else:
        # All parameters known
        D_adjusted = D_raw * (np.sqrt(n) + 0.12 + (0.11 / np.sqrt(n)))
        critical_value = 1.358

    return D_adjusted, critical_value


# ---------------------------------------------------------
# 3. Main loop
# ---------------------------------------------------------

for sheet in sheets:
    try:
        data = read_column(sheet)
        data_np = np.array(data)
        english_name = name_map.get(sheet, sheet)

        # Create two side-by-side plots:
        # Left: histogram, Right: Q-Q plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1 (left): Histogram
        x_axis = np.linspace(np.min(data_np), np.max(data_np), 1000)
        axes[0].hist(data_np, bins=20, edgecolor="black", density=True, alpha=0.5, label="Data Histogram")
        axes[0].set_title(f"Histogram – {english_name}")
        axes[0].set_xlabel("time_minutes")
        axes[0].set_ylabel("frequency")

        print(f"\n{'='*60}")
        print(f"--- ניתוח ידני מלא (MLE + KS Adjusted) עבור: {sheet} ---")

        if sheet == "מגלשת אבובים גדולה":
            # 1. MLE Normal
            mu, sigma = calculate_mle_normal(data_np)

            # 2. Plot PDF on Histogram
            pdf = stats.norm.pdf(x_axis, mu, sigma)
            axes[0].plot(x_axis, pdf, 'r-', linewidth=3, label=f'Normal Fit\n$\mu$={mu:.3f}, $\sigma$={sigma:.3f}')

            # 3. Q-Q Plot
            stats.probplot(data_np, dist="norm", sparams=(mu, sigma), plot=axes[1])
            axes[1].set_title(f"Q-Q Plot (Normal) – {english_name}")

            # 4. KS Test (Modified for Normal)
            D_adj, Crit_val = manual_ks_test(data_np, 'norm', (mu, sigma), alpha=0.05)

            print(f"1. תוצאות MLE:")
            print(f"   Mean = {mu:.4f}, Std = {sigma:.4f}")
            print(f"2. תוצאות KS ")
            print(f"   Adjusted D Statistic = {D_adj:.4f}")
            print(f"   Fixed Critical Value (alpha=0.05) = {Crit_val:.4f}")

            if D_adj < Crit_val:
                print(" >> Conclusion: Adjusted D < Critical Value -> Fail to reject H0 (the fit is adequate) ✅")
            else:
                print("  >> Conclusion: Adjusted D > Critical Value -> Reject H0 (the fit is not adequate) ❌")

        elif sheet == "מגלשת אבובים קטנה ":
            # 1. MLE Exponential
            lam, scale = calculate_mle_expon(data_np)

            # 2. Plot PDF on Histogram
            pdf = stats.expon.pdf(x_axis, scale=scale)
            axes[0].plot(x_axis, pdf, 'g-', linewidth=3, label=f'Expon Fit\n$\lambda$={lam:.3f}')

            # 3. Q-Q Plot
            # שים לב: באקספוננציאלי מעבירים (0, scale) ל-sparams
            stats.probplot(data_np, dist="expon", sparams=(0, scale), plot=axes[1])
            axes[1].set_title(f"Q-Q Plot (Exponential) – {english_name}")

            # 4. KS Test (Modified for Exponential)
            D_adj, Crit_val = manual_ks_test(data_np, 'expon', (lam, scale), alpha=0.05)

            print(f"1. תוצאות MLE:")
            print(f"   Lambda = {lam:.4f}")
            print(f"2. תוצאות KS:")
            print(f"   Adjusted D Statistic = {D_adj:.4f}")
            print(f"   Fixed Critical Value (alpha=0.05) = {Crit_val:.4f}")

            if D_adj < Crit_val:
                print("  >> Conclusion: Adjusted D < Critical Value -> Fail to reject H0 (the fit is adequate) ✅")
            else:
                print("  >> Conclusion: Adjusted D > Critical Value -> Reject H0 (the fit is not adequate) ❌")

        axes[0].legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"שגיאה בגיליון '{sheet}': {e}")


#Sampling Algorithms


class SamplingAlgorithms:
    def __init__(self, seed=None):

        #Central sampling class.
        #Supports seeding for Common Random Numbers (CRN) and configurable parameters for alternative scenarios.

        if seed is not None:
            random.seed(seed)

       # Parameters for the "Improved Kitchen" scenario (default: current state)
        self.prob_eat_lunch = 0.70      #  70% of visitors choose to eat lunch
        self.prob_unsatisfied = 0.10    # else

    def set_seed(self, seed):
        #Update the random seed before each replication
        random.seed(seed)

    def set_kitchen_quality(self, improved: bool):
        #Function to update the kitchen quality based on the selected scenario
        if improved:
            self.prob_eat_lunch = 0.85     # Alternative A: 85% of visitors eat lunch
            self.prob_unsatisfied = 0.03   # Alternative A: only 3% of visitors are dissatisfied
        else:
            self.prob_eat_lunch = 0.70     # base senario
            self.prob_unsatisfied = 0.10   # base senario

    # =================================================================
    # 1. Base function
    # =================================================================
    def get_u(self):
        return random.random()

    def uniform(self, a, b):
        return a + (b - a) * self.get_u()

    def uniform_discrete(self, a, b):
        return int(a + (b - a + 1) * self.get_u())

    def exponential(self, lam):
        return -math.log(1 - self.get_u()) / lam

    def exponential_by_mean(self, mean):
        return self.exponential(1 / mean)

    def normal(self, mu, sigma):
        u1 = self.get_u()
        u2 = self.get_u()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mu + (sigma * z)

    # =================================================================
    # 2. Family and children
    # =================================================================
    def get_num_children_family(self):
        u = self.get_u()
        if u <= 0.2: return 1
        elif u <= 0.4: return 2
        elif u <= 0.6: return 3
        elif u <= 0.8: return 4
        else: return 5

    def get_child_age(self):
        return self.uniform(2, 18)

    def get_family_leave_time(self):
        return 16 + 3 * math.sqrt(self.get_u())

    # =================================================================
    # 3. Wave pool (Acceptance-Rejection)
    # =================================================================
    def _wave_pool_pdf(self, x):
        if 0 <= x <= 30: return x / 2700.0
        elif 30 < x <= 50: return ((60 - x) / 2700.0) + (1/30.0)
        elif 50 < x <= 60: return (60 - x) / 2700.0
        else: return 0.0

    def get_wave_pool_duration(self):
        f_max = (30/2700) + (1/30)
        while True:
            y_candidate = self.uniform(0, 60)
            u = self.get_u()
            if u <= (self._wave_pool_pdf(y_candidate) / f_max):
                return y_candidate

    # =================================================================
    # 4. Teen groups
    # =================================================================
    def get_teen_group_size(self):
        u = self.get_u()
        if u <= 0.20: return 2
        elif u <= 0.40: return 3
        elif u <= 0.65: return 4
        elif u <= 0.90: return 5
        else: return 6

    # =================================================================
    # 5. Decisions
    # =================================================================

    def teen_buy_express_decision(self):
        return self.get_u() <= 0.6

    def decides_to_buy_express_entry(self):
        return self.get_u() < 0.25

    def decides_to_eat_lunch(self):
        return self.get_u() <= self.prob_eat_lunch

    def is_food_unsatisfied(self):
        return self.get_u() < self.prob_unsatisfied

    def choose_restaurant(self):
        u = self.get_u()
        if u <= 0.375: return "Burger"
        elif u <= 0.625: return "Pizza"
        else: return "Salad"

    def family_should_split(self):
        return self.get_u() <= 0.6

    def get_num_subgroups(self):
        if self.get_u() <= 0.5: return 2
        return 3


    # =================================================================
    # 6. Toddler_pool
    # =================================================================
    def Toddler_pool(self):
        u = self.get_u()
        if u < 1/6:
            hours = 1 + math.sqrt((3/8) * u)
        elif u < 5/6:
            hours = 1.25 + (3/4) * (u - 1/6)
        else:
            hours = 2 - math.sqrt((3/8) * (1 - u))
        return hours * 60


class Queue:
    def __init__(self):
        self.regular = []
        self.express = []

        self.last_time_minutes = 540 #open time = 540
        self.empty_time_minutes = 0.0 #how much time the queue was empty

        self.total_wait_time_minutes = 0.0
        self.num_served = 0
        self.num_wait_observations = 0


    def _update_time_stats(self, now_minutes):
        dt = now_minutes - self.last_time_minutes
        if dt < 0:
            dt = 0

        # the queue is considered empy if there is nobody in the queue
        if self.size_people() == 0:
            self.empty_time_minutes += dt

        self.last_time_minutes = now_minutes

    def add(self, VisitorGroup, now_minutes): #adding groups to queue
        self._update_time_stats(now_minutes)

        if VisitorGroup.has_express_band:
            self.express.append((VisitorGroup, now_minutes))
        else:
            self.regular.append(( VisitorGroup, now_minutes))

    def promote_to_front(self, group, now_minutes): #Specialized method for the snorkel ride, enabling a group to be moved to the front of the queue
        # Update statistics
        self._update_time_stats(now_minutes)

        # # Select queue based on express band
        if group.has_express_band:
            q = self.express
        else:
            q = self.regular

         # Insert directly to the front of the queue
        q.insert(0, (group, now_minutes))

        return True


    def remove_next(self, now_minutes, idx=0, from_express=True):

        self._update_time_stats(now_minutes)

        # Select which queue to remove the next group from
        if from_express and len(self.express) > 0:
            q = self.express
        elif len(self.regular) > 0:
            q = self.regular
        elif len(self.express) > 0:   # If regular was requested but is empty and express is not
            q = self.express
        else:
            return None

        if idx < 0 or idx >= len(q):
            return None

        group, enter_time = q.pop(idx)


        wait = now_minutes - enter_time
        if wait < 0:
            wait = 0

        self.total_wait_time_minutes += wait
        self.num_served += 1
        group.add_wait_time(wait)
        self.num_wait_observations += 1

        return group

    def abandon_group(self, group, now_minutes, rating_penalty=0.8):
        self._update_time_stats(now_minutes)

        # If the group is in the express queue — abandonment is not allowed
        for (g, _) in self.express:
            if g is group or g.id == group.id:
                return None

        # Abandonment is allowed only from the regular queue
        for i, (g, enter_time) in enumerate(self.regular):
            if g is group or g.id == group.id:
                self.regular.pop(i)
                wait = max(0, now_minutes - enter_time)
                self.total_wait_time_minutes += wait
                g.add_wait_time(wait)
                self.num_wait_observations += 1
                g.update_rating(-rating_penalty)
                return g

        return None


    def empty(self): # Checks whether the queue is empty
        return self.size_people() == 0

    def size_people(self): # Returns the total queue size in terms of people, not groups
        total = 0
        for VisitorGroup, _ in self.regular:
            total += VisitorGroup.size
        for VisitorGroup, _ in self.express:
            total += VisitorGroup.size
        return total

    def average_wait_time_minutes(self):
        if self.num_wait_observations == 0:
            return 0.0
        return self.total_wait_time_minutes / self.num_wait_observations


    def evacuate_all(self, now_minutes, rating_penalty=0.0):
        self._update_time_stats(now_minutes)
        evacuated = []

        for g, enter_time in self.express:
            wait = now_minutes - enter_time
            if wait < 0: wait = 0
            self.total_wait_time_minutes += wait
            g.add_wait_time(wait)
            self.num_wait_observations += 1
            if rating_penalty != 0.0:
                g.update_rating(-rating_penalty)
            evacuated.append(g)

        for g, enter_time in self.regular:
            wait = now_minutes - enter_time
            if wait < 0: wait = 0
            self.total_wait_time_minutes += wait
            g.add_wait_time(wait)
            self.num_wait_observations += 1
            if rating_penalty != 0.0:
                g.update_rating(-rating_penalty)
            evacuated.append(g)

        self.express.clear()
        self.regular.clear()
        self._update_time_stats(now_minutes)
        return evacuated


    def remove_group_everywhere(self, group, now_minutes, rating_penalty=0.0): # Remove a group from either regular or express queue (used for park exit)

      self._update_time_stats(now_minutes)

      # Try express queue first
      for i, (g, enter_time) in enumerate(self.express):
          if g is group or g.id == group.id:
              self.express.pop(i)

              wait = max(0, now_minutes - enter_time)
              self.total_wait_time_minutes += wait
              g.add_wait_time(wait)
              self.num_wait_observations += 1

              if rating_penalty != 0.0:
                  g.update_rating(-rating_penalty)

              return True

      # Then try regular queue
      for i, (g, enter_time) in enumerate(self.regular):
          if g is group or g.id == group.id:
              self.regular.pop(i)

              wait = max(0, now_minutes - enter_time)
              self.total_wait_time_minutes += wait
              g.add_wait_time(wait)
              self.num_wait_observations += 1

              if rating_penalty != 0.0:
                  g.update_rating(-rating_penalty)

              return True

      return False

"""# Group Class"""

class VisitorGroup:
    def __init__(self, group_id, size, arrival_time, initial_rating=10.0):
        # ===========
        #  Identity
        # ===========
        self.id = group_id
        self.size = size
        self.arrival_time = arrival_time
        self.type_name = "General"
        self.ages = [20] * size  # Default ages (overwritten by subclasses)

        # =========
        #  Status
        # =========
        self.current_rating = initial_rating
        self.patience_limit = 0.0
        self.has_express_band = False
        self.finished_visits = False
        self.has_eaten_lunch = False
        self.visited_activities = [] # Log of visited activity IDs

        # =================
        #  Time Management
        # =================
        # Timers indicating when the group becomes free
        self.ride_until = None
        self.food_payment_until = None
        self.eating_until = None

        # ===============================
        #  Split & Subgroup Management
        # ===============================
        self.parent_ref = None
        self.members_outside = 0
        self.active_subgroups_refs = []
        self.temp_ages_collection = []
        self.wants_to_leave = False

        # ==========================================
        #  Abandonment & Retry Logic (Teens)
        # ==========================================
        self.has_started_ride = False
        self.abandoned_history = set()
        # Fields for smart retry behavior (leaving queue to buy express and return)
        self.retry_activity_id = None
        self.skip_retry_once = False

        # ===============
        # 6. Statistics
        # ===============
        self.stats = {
            'money_spent': 0.0,
            'total_wait_time': 0.0,
            'facilities_visited': 0,
        }
    # ===============
    # Core Methods
    # ===============
   # Checks if the group is currently busy.
   # Returns the timestamp when they become free, or None.
    def busy_until(self):
      times = [
        self.ride_until,
        self.food_payment_until,
        self.eating_until
      ]
      times = [t for t in times if t is not None]
      return max(times) if times else None

    # Splits the current group into a subgroup.
    # Handles both Toddler Pool splits (by age) and Capacity splits (by size).
    def create_chunks(self, size_to_take, toddler_pool=False):

        # --- Determine who goes and who stays ---
        if toddler_pool:
          # Split by age for Toddler Pool (ages <= 4)
            toddlers = []
            others = []
            for age in self.ages:
                if age <= 4:
                    toddlers.append(age)
                else:
                    others.append(age)

            if not toddlers:
                return None # No eligible members

            chunk_ages = toddlers
            remaining_ages = others
            current_chunk_size = len(toddlers)

        # Split by capacity limit
        else:
            chunk_ages = self.ages[:size_to_take]
            remaining_ages = self.ages[size_to_take:]
            current_chunk_size = size_to_take

        # --- Update Parent Group ---
        self.ages = remaining_ages
        self.size = len(remaining_ages)
        self.members_outside += current_chunk_size

       # --- Create New Subgroup ---
        chunk_group = VisitorGroup(f"{self.id}_chunk", current_chunk_size, self.arrival_time, initial_rating=self.current_rating)

       # Transfer State & Attributes
        chunk_group.parent_ref = self
        chunk_group.ages = chunk_ages
        chunk_group.type_name = self.type_name
        chunk_group.has_express_band = self.has_express_band
        chunk_group.patience_limit = self.patience_limit
        chunk_group.has_eaten_lunch = self.has_eaten_lunch

        # Initialize Stats
        chunk_group.stats = {
            'money_spent': 0.0,
            'total_wait_time': 0.0,
            'facilities_visited': 0,
            'total_time': 0.0
        }

        # Copy History & Logic Flags
        chunk_group.visited_activities = self.visited_activities.copy()
        chunk_group.abandoned_history = self.abandoned_history.copy()
        chunk_group.retry_activity_id = self.retry_activity_id
        chunk_group.skip_retry_once = self.skip_retry_once

        # Track the new subgroup
        self.active_subgroups_refs.append(chunk_group)
        return chunk_group

    # Merges a returning subgroup back into the parent.
    # Aggregates statistics and checks for full reunion.
    def receive_chunk(self, returning_chunk):
      # Remove from active list
        if returning_chunk in self.active_subgroups_refs:
            self.active_subgroups_refs.remove(returning_chunk)

        self.temp_ages_collection.extend(returning_chunk.ages)
        self.members_outside -= returning_chunk.size

        # --- Aggregate Statistics ---
        self.stats['total_wait_time'] = max(self.stats['total_wait_time'], returning_chunk.stats['total_wait_time'])
        self.stats['money_spent'] += returning_chunk.stats['money_spent']
        self.stats['facilities_visited'] += returning_chunk.stats['facilities_visited']

        # Merge visited history
        for act_id in returning_chunk.visited_activities:
            if act_id not in self.visited_activities:
                self.visited_activities.append(act_id)

        # Check if fully reunited
        if self.members_outside == 0:
            self.ages.extend(self.temp_ages_collection)
            self.size = len(self.ages)
            self.temp_ages_collection = []
            self.has_started_ride = False # Reset ride status

            if self.wants_to_leave:
               return "LEAVE_NOW"

            return True
        return False

    # Forced collection of all subgroups (at the end of simulation).
    # Removes them from queues or waits for rides to finish.
    def collect_active_subgroups(self, simulation, current_time):
        if self.members_outside == 0:
            return None
        max_busy_time = 0

        # Iterate over a copy to allow modification of the original list
        for subgroup in list(self.active_subgroups_refs):
            found_and_removed = False

            # Try removing from Activity Queues
            for activity in simulation.activities:
               if activity.queue.remove_group_everywhere(subgroup, current_time, rating_penalty=0.0):
                    self.receive_chunk(subgroup)
                    found_and_removed = True
                    break

            # Try removing from Food Queues
            if not found_and_removed:
                for stall in simulation.food_court.stalls.values():
                    if stall.queue.remove_group_everywhere(subgroup, current_time, rating_penalty=0.0):
                        self.receive_chunk(subgroup)
                        found_and_removed = True
                        break

            # If in a ride/service, check finish time
            if not found_and_removed:
                bt = subgroup.busy_until()
                if bt and bt > max_busy_time:
                    max_busy_time = bt

        if self.members_outside == 0:
            return None

        # Return the time when the last subgroup finishes
        return max_busy_time if max_busy_time > current_time else current_time + 1.0

      # Forcefully aggregates statistics from all active subgroups.
      # Used as a fallback mechanism at the end of the simulation to ensure
      # no data is lost for groups that are still split.
    def force_aggregate_statistics(self, current_sim_time):
      actual_leave_time = current_sim_time

      if self.members_outside > 0:
          # Iterate over active subgroups to capture their final state
          for subgroup in self.active_subgroups_refs:

              # 1. Update Wait Time (Max logic: the group waited as long as the member who waited the most)
              if subgroup.stats['total_wait_time'] > self.stats['total_wait_time']:
                  self.stats['total_wait_time'] = subgroup.stats['total_wait_time']

              # 2. Update Money Spent
              # Note: Restoring original logic (Max instead of Sum).
              if subgroup.stats['money_spent'] > self.stats['money_spent']:
                  self.stats['money_spent'] = subgroup.stats['money_spent']

              # 3. Check for realistic finish time (if a child is stuck in a ride)
              kid_finish_time = subgroup.busy_until()
              if kid_finish_time is not None:
                  if kid_finish_time > actual_leave_time:
                      actual_leave_time = kid_finish_time

          # Clear the subgroups list as we have processed them
          self.members_outside = 0
          self.active_subgroups_refs = []

      # Check if the parent itself is busy until a later time
      my_finish_time = self.busy_until()
      if my_finish_time is not None:
          if my_finish_time > actual_leave_time:
              actual_leave_time = my_finish_time

      return actual_leave_time

  # ===================
  # Helpers & Setters
  # ===================
  # Checks if the group meets the requirements to enter the activity.
  # Base implementation checks only age constraints.
    def can_ride(self, activity):
        for age in self.ages:
            if age < activity.min_age:
                return False
        return True

    def mark_visited(self, activity_id):
        self.visited_activities.append(activity_id)
        self.record_visit()

    def has_finished_park(self):
        target = 6
        if self.type_name == "Teen":
           target = 4

        if len(self.visited_activities) >= target:
            self.wants_to_leave = True
            return True
        return False

    def has_visited(self, activity_id):
        return activity_id in self.visited_activities

    def add_wait_time(self, minutes):
        self.stats['total_wait_time'] += minutes

    def record_spending(self, amount):
        self.stats['money_spent'] += amount
        return True

    def buy_express_band(self):
        if not self.has_express_band:
            cost = 50 * self.size
            self.record_spending(cost)
            self.has_express_band = True
            return True
        return False

    def record_visit(self):
        self.stats['facilities_visited'] += 1

    def update_rating(self, change):
        self.current_rating += change
        if self.current_rating < 0.0: self.current_rating = 0.0
        elif self.current_rating > 15.0: self.current_rating = 15.0

    # default behavior (returning empty/False) for all visitor types.
    # Specific subclasses (Family\ TeenGroup) override these methods to implement their unique logic.
    def try_split(self, sampler): return []
    def check_patience_decision(self, current_wait_time, sampler): return False

    # Displays the object as "[Type ID]"
    def __repr__(self): return f"[{self.type_name} {self.id}]"

# --- Families ---
class Family(VisitorGroup):
    def __init__(self, group_id, size, arrival_time, ages_list, initial_rating=10.0):
        # Initialize parent class (inherits statistics, chunk logic, parent reference, etc.)
        super().__init__(group_id, size, arrival_time, initial_rating)

        self.type_name = "Family"
        self.patience_limit = 15.0  # Families are willing to wait up to 15 minutes

        # Store the actual ages sampled by the Sampler
        self.ages = ages_list

        # Helper calculation: number of responsible members (age >= 12)
        self.num_over_12 = sum(1 for age in self.ages if age >= 12)

        # Flag indicating whether the family has already attempted to split
        self.has_attempted_split = False

    def try_split(self, sampler): #The split is only attempted if the family is large enough

        # 1. Basic threshold checks
        if self.size < 3:
            return []  # Too small to split
        self.has_attempted_split = True
        # Decide whether the family wants to split (probability = 0.6)
        if not sampler.family_should_split():
          return []

        # 2. Determine the number of subgroups (2 or 3)
        num_subgroups = sampler.get_num_subgroups()

        # Safety check: cannot create more subgroups than members
        if self.size < num_subgroups:
            num_subgroups = self.size

        # 3. Feasibility check:
        if self.num_over_12 < num_subgroups:
            return []

         # --- Age distribution logic

        # Separate members into adults and children
        adults = [a for a in self.ages if a >= 12]
        kids = [a for a in self.ages if a < 12]

        # Prepare containers for the new subgroups
        new_ages_lists = [[] for _ in range(num_subgroups)]

        current_group_index = 0

        # a. Distribute adults first
        for adult in adults:
            new_ages_lists[current_group_index].append(adult)
            current_group_index += 1
            if current_group_index == num_subgroups:
                current_group_index = 0

        # b. Distribute children
        for kid in kids:
            new_ages_lists[current_group_index].append(kid)
            current_group_index += 1
            if current_group_index == num_subgroups:
                current_group_index = 0

         # --- Create the new subgroup objects
        new_groups = []
        for i in range(num_subgroups):
            ages = new_ages_lists[i]
            sub_size = len(ages)
            # Create a temporary unique ID
            sub_id = f"{self.id}-{i+1}"

            # Create the new Family subgroup
            sub_group = Family(sub_id, sub_size, self.arrival_time, ages, initial_rating=self.current_rating)

            # ==========================================
            # Critical updates for future reunification
            # ==========================================

            # 1. Preserve original family identity
            sub_group.parent_id = self.id

            # 2. Store total number of sibling subgroups
            sub_group.total_siblings = num_subgroups

            # 3. Copy history so money spent and waiting time are preserved
            sub_group.stats = self.stats.copy()
            sub_group.visited_activities = self.visited_activities.copy()

            new_groups.append(sub_group)
            self.active_subgroups_refs.append(sub_group)
            self.members_outside += sub_group.size

        return new_groups

# --- Teen Groups
class TeenGroup(VisitorGroup):
    def __init__(self, group_id, size, arrival_time, initial_rating=10.0):
        super().__init__(group_id, size, arrival_time, initial_rating)
        self.type_name = "Teen"
        self.patience_limit = 20.0
        # All teens are assumed to be 16 years old
        self.ages = [16] * size
        # Fields for smart retry behavior
        self.retry_activity_id = None
        self.skip_retry_once = False

    def check_patience_decision(self, current_wait_time, sampler):
        if sampler.teen_buy_express_decision():
            self.buy_express_band()
            return True
        return False

    def can_ride(self, activity):
        """
        Teens have specific preferences:
        1. Must meet age requirements (Base check).
        2. Must be an adrenaline ride (Level >= 3).
        """
        # 1. Check basic age requirements (using parent logic)
        if not super().can_ride(activity):
            return False

        # 2. Check Adrenaline preference (Teens only go to 3+ waves)
        if activity.adrenaline < 3:
            return False

        return True
# -----Singles---------
class Single(VisitorGroup):
    def __init__(self, group_id, arrival_time, initial_rating=10.0):
        super().__init__(group_id, size=1, arrival_time=arrival_time, initial_rating=initial_rating)
        self.type_name = "Single"
        self.patience_limit = 30.0
        # single is an adult
        self.ages = [25]

"""#Activity Class"""

# ------------------------------
#       Class representing an activity (ride)
# ------------------------------
class Activity:
  def __init__(
    self,
    simulation: "Simulation",
    activity_id: int,
    name: str,
    min_age: int,
    adrenaline: int,
    unit_capacity: int, #number of people per unit
    max_units_parallel: int, #number of units that can operate in parallel
    duration_sampler: Callable[[], float], #sample activity duration
    queue: Optional["Queue"] = None, ##queue instance
  ):

    #---- link to simulation-------
    self.simulation = simulation

    # --- Basic info ---
    self.activity_id = activity_id
    self.name = name
    self.min_age = min_age
    self.adrenaline = adrenaline
    self.unit_capacity = unit_capacity
    self.max_units_parallel = max_units_parallel
    self.duration_sampler = duration_sampler

    # --- Queue & units ---
    self.queue = Queue() if queue is None else queue

    #------servers--------
    self.is_busy = [False] * self.max_units_parallel      # False = idle, True = busy (one entry per server)
    self.server_busy_until = [0.0] * self.max_units_parallel    # when service ends for each server
    self.server_last_change = [0.0] * self.max_units_parallel   # last state-change time (for statistics)

    # current content of each server
    self.server_groups = [None] * self.max_units_parallel   # which groups are currently on the server
    self.server_start_time = [0.0] * self.max_units_parallel # when service started
    self.server_num_people = [0] * self.max_units_parallel # number of people currently on the server


    # --- Stats ---
    self.total_units_started = 0
    self.total_people_served = 0
    self.total_service_time = 0.0
    self.num_abandon = 0

    #---------snorkeling----------
    # --------- lunch break logic----------
    self.instructor_on_lunch = [False] * self.max_units_parallel


    ##--- single slide-----
    self.min_entry_gap = 0.0   # default is 0 minutes
    self.next_entry_time = [-float("inf")] * self.max_units_parallel  # when the next slide entry is allowed
    self.inflight = [[] for _ in range(self.max_units_parallel)] # riders currently sliding, per slide


  # ---------- add group to queue----------
  def add_to_queue(self, group, now_minutes: float):

    # special handling for toddler pool
    if self.activity_id == 6:
      chunk = group.create_chunks(0, toddler_pool=True) # Size irrelevant, take all toddlers
      if chunk is None:
        return False
      self.queue.add(chunk, now_minutes) # Only toddlers enter, adults wait outside

      abandon_time = now_minutes + chunk.patience_limit
      self.simulation.schedule_event(
        AbandonmentEvent(abandon_time, chunk, self)
      )

      self.try_start_units(now_minutes)
      return True
    if not self.check_min_age(group, self.min_age): ##minimun age check
      return False

    if group.has_visited(self.activity_id): ##prevent repeated visit to same activity
      return False

    if group.size > self.unit_capacity: ##handle groups larger that capacity
      groups = self._split_for_capacity(group) ##list of sub groups
      for g in groups:
        self.add_to_queue(g, now_minutes)
      self.try_start_units(now_minutes)
      return True
    else:
      self.queue.add(group, now_minutes)

      ##abandment event
      patience = group.patience_limit
      abandon_time = now_minutes + patience

      self.simulation.schedule_event(
        AbandonmentEvent(time=abandon_time, activity=self, group=group)
      )

      self.try_start_units(now_minutes)

      return True



  # -------------check if minimum age in the group fitt the activity---------
  def check_min_age(self, group, minAge):
    for i in group.ages:
      if i < minAge:
        return False
    return True


  ##-----------split group because of activity capacity----------
  def _split_for_capacity(self, group):
    cap = self.unit_capacity

    chunk_sizes = [] ## how many group do I need?
    remaining = group.size
    while remaining > 0:
      chunk = min(cap, remaining)
      chunk_sizes.append(chunk)
      remaining -= chunk

    # creat sub groups
    chunks = []
    for sz in chunk_sizes:
      chunk = group.create_chunks(sz)
      chunks.append(chunk)

    return chunks


  # ------------ Fill one unit using "maximum people" policy  ----------
  def fill_one_unit(self, now_minutes):
    assigned = [] ##groups assigned to this round
    remaining = self.unit_capacity ##how many people can still join this round?

    while remaining > 0:
      chosen_from_express = True
      chosen_idx = None

      #  try express queue first
      for i in range(len(self.queue.express)):
        group, enter_time = self.queue.express[i]
        if group is None:
          continue
        if group.size <= remaining:
          chosen_from_express = True
          chosen_idx = i
          break



      # 2) if not found, try regular queue
      if chosen_idx is None:
        chosen_from_express = False
        for i in range(len(self.queue.regular)):
          group, enter_time = self.queue.regular[i]
          if group is None:
            continue
          if group.size <= remaining:
            chosen_idx = i
            break

      # 3) no suitable group found - stop
      if chosen_idx is None:
        break

      group = self.queue.remove_next(now_minutes, chosen_idx, chosen_from_express) ## the chosen group

      if group is None:
        break
      assigned.append(group)
      remaining -= group.size

    return assigned


  # ----------release finish server------------
  def release_finished_servers(self, server_idx: int, now_minutes: float):

    if server_idx < 0 or server_idx >= self.max_units_parallel: #basic bounds check
      return

    if self.activity_id == 2: ##single slide
      self._handle_single_slide_finish(server_idx, now_minutes)
      return


     # if server is idle
    if not self.is_busy[server_idx]:
      return

    #
    finished_groups = self.server_groups[server_idx]


    # snorkeling lunch break logic
    if self.activity_id == 7:
      if self.server_groups[server_idx] is None: #returning from lunch
        self.is_busy[server_idx] = False
        self.server_last_change[server_idx] = now_minutes
        self.try_start_units(now_minutes)
        return

      else: #going to lunch
        self.is_busy[server_idx] = True
        self.server_groups[server_idx] = None
        self.server_num_people[server_idx] = 0
        self.server_start_time[server_idx] = 0.0

        finish_event = FinishUnitEvent(
          time=now_minutes + 30,
          activity=self,
          server_index= server_idx
        )
        self.simulation.schedule_event(finish_event)
        self._handle_finished_groups(finished_groups, now_minutes)
        return


    else:
      self.is_busy[server_idx] = False
    self.server_groups[server_idx] = None
    self.server_num_people[server_idx] = 0
    self.server_start_time[server_idx] = 0.0
    self.server_last_change[server_idx] = now_minutes


    self._handle_finished_groups(finished_groups, now_minutes)



  # ------------- try starting new units----------
  def try_start_units(self, current_time: float):
    # single slide
    if self.activity_id == 2:
        self._try_start_single_slide_units(current_time)
        return

    for i in range(self.max_units_parallel):

      # skip busy server
      if self.is_busy[i]:
          continue

      # skip if instructor on lunch
      if self.instructor_on_lunch[i]:
        continue

      # no one waiting in queue
      if self.queue.empty():
          break

      actual_start_time = current_time

      assigned_groups = self.fill_one_unit(current_time)
      if not assigned_groups:
          break

      for g in assigned_groups:
        if g.parent_ref:
          g.parent_ref.has_started_ride = True


      duration = self.duration_sampler()
      end_time = actual_start_time + duration

      #  check if ride finish before park closing time
      if end_time > self.simulation.close_time:
        for g in assigned_groups:
          g.ride_until = None
          # if this is a sub group
          if g.parent_ref is not None:
            parent = g.parent_ref
            parent.receive_chunk(g)
        self.try_start_units(current_time)
        break

      #snorkeling lunch constraints
      if self.activity_id == 7:
        lunch_start = self.simulation.snorkel_lunch_start
        lunch_end = self.simulation.snorkel_lunch_end

        # if in lunch window- do not start at all
        if lunch_start <= current_time < lunch_end:
          #return groups to queue cause they were already removed in fill one unit
          for g in assigned_groups:
            self.queue.promote_to_front(g, current_time)
            g.ride_until = None
          break

        # if the ride will end over into lunch time- do not start
        if current_time < lunch_start and end_time > lunch_start:
          for g in assigned_groups:
            self.queue.promote_to_front(g, current_time)
            g.ride_until = None
          break

      for g in assigned_groups:
        g.ride_until = end_time
        if g.parent_ref is not None:
            g.parent_ref.ride_until = end_time

      num_people = sum(g.size for g in assigned_groups)

      # activate server
      self.is_busy[i] = True

      self.server_busy_until[i] = end_time
      self.server_groups[i] = assigned_groups
      self.server_start_time[i] = actual_start_time
      self.server_num_people[i] = num_people
      self.server_last_change[i] = current_time


      # statistics
      #self.total_units_started += 1
      #self.total_people_served += num_people
      #self.total_service_time += duration

      finish_event = FinishUnitEvent(
        time=end_time,
        activity=self,
        server_index=i
      )
      self.simulation.schedule_event(finish_event)


  ##-------------functions for handling single slide--------------

  def _try_start_single_slide_units(self, current_time): ##actually inserts people into the single slide


      for i in range(self.max_units_parallel): ##uterate over all slides

          if current_time < self.next_entry_time[i]: ##entry not allowed yet (30 seconds)
              continue

          if self.queue.empty(): ##if there is no one in the queue
              break


          assigned = self.fill_one_unit(current_time)  #group of one person according to the capacity
          if not assigned:
              break

          g = assigned[0]
          ride_end = current_time + 3.0

          #can we finish the slide before park is close?
          if ride_end > self.simulation.close_time:
            for g in assigned:
              g.ride_until = None
              # if this is a sub group
              if g.parent_ref is not None:
                parent = g.parent_ref
                parent.receive_chunk(g)
            self.try_start_units(current_time)
            break


          g.ride_until = ride_end
          if g.parent_ref is not None:
            g.parent_ref.ride_until = ride_end

          self.inflight[i].append((g, ride_end)) ##add rider to inflight list
          self.next_entry_time[i] = current_time + self.min_entry_gap ##next antry time in 30 seconds

          # statistics
          #self.total_units_started += 1
          #self.total_people_served += g.size
          #self.total_service_time += 3.0



          # 30 seconds tick- only if there is someone in the queue
          if not self.queue.empty():
              self.simulation.schedule_event(
                  FinishUnitEvent(time=self.next_entry_time[i], activity=self, server_index=i)
              )


          # finish event in 3 minutes
          self.simulation.schedule_event(
              FinishUnitEvent(time=ride_end, activity=self, server_index=i)
          )

  def _handle_single_slide_finish(self, server_idx: int, now_minutes: float):
      finished = [] #finish riders list
      still = [] #still ride list

      for (g, end_t) in self.inflight[server_idx]: ##sort into the two of the lists
          if end_t <= now_minutes:
              finished.append(g)
          else:
              still.append((g, end_t))

      self.inflight[server_idx] = still


      if finished: #send the finish riders to the regular finish event
          self._handle_finished_groups(finished, now_minutes)



      # If this is just a 30-second tick, do not wake the system unnecessarily
      #only if there is true event and there is queue
      if not finished:
          if (not self.queue.empty()) and (now_minutes >= self.next_entry_time[server_idx]):
              self._try_start_single_slide_units(now_minutes)
          return

      # wake the system up
      self._try_start_single_slide_units(now_minutes)



  def _handle_finished_groups(self, finished_groups, now):
    for g in finished_groups:
      g.ride_until = None
      # if this is a sub group
      if g.parent_ref is not None:
        parent = g.parent_ref
        parent.receive_chunk(g)

        # complete the ride for the parent only if this was the last child
        if parent.members_outside != 0:
          continue

        target = parent
      else:
        target = g
      target.ride_until = None

      target.mark_visited(self.activity_id)

      GS = target.size
      A = self.adrenaline

      if self.simulation.sampling.get_u() > 0.5:
        delta = self.good_experience_value(GS, A)
        target.update_rating(delta)
      else:
        target.update_rating(-0.1)


      # sends to lunch
      if 13*60 <= now <= 15*60 and (not target.has_eaten_lunch):
        # choose to eat 70%
        if self.simulation.food_court.decide_to_eat():
          self.simulation.send_group_to_food(target, now)

          continue  #do not continue to the next activity




      if target.type_name != 'Family':
        # if they completed enough visits, leave now
        if target.has_finished_park():
          self.simulation.schedule_event(LeaveParkEvent(now + 1, target, reason="FINISHED_6"))
        else:
          # otherwise continue to the next activity (or leave if none exists)
          self.simulation.send_group_to_next_activity(target, now)

      else:
        self.simulation.send_group_to_next_activity(target, now)



  def good_experience_value(self, GS, A):
    score = ((GS - 1) / 5) * 0.3 + ((A - 1) / 4) * 0.7
    return score

"""#FoodCourt + FoodStall Classes"""

class FoodStall:
    def __init__(self, name: str, sampler: SamplingAlgorithms):
        self.name = name
        self.sampler = sampler
        self.queue = Queue()
        self.server_busy = False      # if the server is busy

        # for statistics
        self.num_served = 0
        self.total_service_time = 0.0
        self.revenue = 0.0

    def add_to_queue(self, group: VisitorGroup, now_minutes: float):
        #there is no express queue
        self.queue._update_time_stats(now_minutes)
        self.queue.regular.append((group, now_minutes))

    def can_start_service(self) -> bool:
        return (not self.server_busy) and (not self.queue.empty())

    def _positive_normal(self, mu, sigma):
          x = self.sampler.normal(mu, sigma)
          while x <= 0:
             x = self.sampler.normal(mu, sigma)
          return x

    def start_service(self, now_minutes: float):
        if not self.can_start_service():
            return None
        self.server_busy = True

        # Remove the next group from the queue
        # (the Queue object also computes waiting time and updates the group's statistics)
        group = self.queue.remove_next(now_minutes, from_express=False)
        if group is None:
            self.server_busy = False
            return None

        service_time = self._positive_normal(mu=5.0, sigma=1.5)  # Service time at the stall

        if self.name == "Burger":
            prep_time = self.sampler.uniform(3, 4)
        elif self.name == "Pizza":
            prep_time = self.sampler.uniform(4, 6)
        elif self.name == "Salad":
            prep_time = self.sampler.uniform(3, 7)
        else:
            prep_time = 0.0

        total_time = service_time + prep_time

        self.num_served += 1
        self.total_service_time +=  total_time

        return group, now_minutes + total_time

    def finish_service(self, group: VisitorGroup): # Finish service: free the stall and charge the group.

        self.server_busy = False

        price = self._calc_price(group)
        group.record_spending(price)
        self.revenue += price


        return price

    def _calc_price(self, group: VisitorGroup) -> float:

        if self.name == "Burger":
            return 100
        if self.name == "Salad":
            return 65

        if self.name == "Pizza":
          if group.size == 1:
              return 40
          else:
            return 100

        return 0.0


# FoodCourt = the food court area (decides whether to eat + which stall + eating time)

class FoodCourt:
    def __init__(self, sampler: SamplingAlgorithms):
        self.sampler = sampler

        # Three stalls stored in a dictionary by name
        self.stalls = {"Burger": FoodStall("Burger", sampler),"Pizza":  FoodStall("Pizza",  sampler),"Salad":  FoodStall("Salad",  sampler)}

    def is_lunch_time(self, now_minutes) -> bool:

        return (13 * 60) <= now_minutes <= (15 * 60)

    def decide_to_eat(self) -> bool:
        return self.sampler.decides_to_eat_lunch()

    def choose_stall_name(self) -> str:
        return self.sampler.choose_restaurant()  # "Burger"/"Pizza"/"Salad"

    def eating_duration(self) -> float:
        return self.sampler.uniform(15, 35)

    def maybe_unsatisfied(self, group: VisitorGroup) -> bool:
        if self.sampler.is_food_unsatisfied():
            group.update_rating(-0.8)
            return True
        return False

"""#EntranceStation Class"""

class EntranceStation:
    def __init__(self, sampler, num_servers=3, has_website=False):
        self.sampler = sampler
        self.num_servers = num_servers
        self.busy = 0 #servers are free
        self.queue = Queue()
        self.has_website = has_website

        self.total_ticket_revenue = 0.0
        self.total_express_revenue = 0.0

    def enqueue(self, group, now_minutes):
        # there is no express queue
        self.queue._update_time_stats(now_minutes)
        self.queue.regular.append((group, now_minutes))

    def can_start(self):
        return (self.busy < self.num_servers) and (not self.queue.empty())

    def start_service(self, now_minutes):
        if not self.can_start():
            return None

        self.busy += 1

        group = self.queue.remove_next(now_minutes, from_express=False)
        if group is None:
            self.busy -= 1
            return None

        if self.has_website:
            t_ticket = 0.0
        else:
            t_ticket = self.sampler.uniform(0.5, 2)
        t_band = self.sampler.exponential_by_mean(2)

        if self.sampler.decides_to_buy_express_entry():
            if group.buy_express_band():  # charging the group
                self.total_express_revenue += 50 * group.size


        service_time = t_ticket + t_band

        return group, now_minutes + service_time

    def finish_service(self, group):
        self.busy -= 1
        if self.busy < 0:
            self.busy = 0

        ticket_cost = self._calc_ticket_price(group)
        group.record_spending(ticket_cost)
        self.total_ticket_revenue += ticket_cost
        group.entered_park = True


    def _calc_ticket_price(self, group):
        total = 0
        for age in group.ages:
            if age >= 14:
                total += 150
            elif age >= 2:
                total += 75
        return total

"""#Events Classes"""

class Event:

    def __init__(self, time: float):
        self.time = time

    def handle(self, simulation):
        raise NotImplementedError("Event subclasses must implement handle()")

    def __lt__(self, other):
        # Allows comparison between events based on their time, so the event list (priority queue) can sort them chronologically
        return self.time < other.time

class FamilyArrivalEvent(Event):
    def handle(self, simulation):
        if simulation.current_time >= simulation.close_time:
            return

        simulation.handle_family_arrival()

        lam_family = 40.0 / 60.0
        next_time = simulation.current_time + simulation.sampling.exponential(lam_family)

        #  family arrive until 12:00
        if next_time <= 12*60 and next_time <= simulation.close_time:
            simulation.schedule_event(FamilyArrivalEvent(next_time))

class TeensArrivalEvent(Event):
    def handle(self, simulation):
        if simulation.current_time >= simulation.close_time:
            return

        simulation.handle_teens_arrival()

        lam_teens = 500.0 / (6 * 60.0)
        next_time = simulation.current_time + simulation.sampling.exponential(lam_teens)

        # Teens arrive before 16:00
        if next_time <= 16*60 and next_time <= simulation.close_time:
            simulation.schedule_event(TeensArrivalEvent(next_time))


class SingleArrivalEvent(Event):
    def handle(self, simulation):
        if simulation.current_time >= simulation.close_time:
            return

        simulation.handle_single_arrival()

        lam_single = 10.0 / 15.0
        next_time = simulation.current_time + simulation.sampling.exponential(lam_single)

        # Singles: arrive up to half an hour before closing.
        if next_time <= (simulation.close_time - 30):
            simulation.schedule_event(SingleArrivalEvent(next_time))

class FinishEntranceEvent(Event):
    def __init__(self, time, group):
        super().__init__(time)
        self.group = group

    def handle(self, simulation):
        # 1. Release the server at the entrance station (update statistics and the busy counter).
        simulation.entrance_station.finish_service(self.group)

        # 2. Send the group to the next activity
        # If the ride is available, it will generate a FinishUnitEvent.
        simulation.send_group_to_next_activity(self.group, self.time)

        # start service for the next group in the entrance queue.
        # This function is responsible for creating the next FinishEntranceEvent and inserting it into the event calendar.
        simulation.try_start_entrance_services(self.time)

class AbandonmentEvent(Event):
    def __init__(self, time, group, activity):
        super().__init__(time)
        self.group = group
        self.activity = activity
        self.PENALTY = 0.8   # abandonment penalty = -0.8

    def handle(self, simulation):
        # -------------------------------------------------
        # Protection for split groups (trapped/stuck groups)
        # -------------------------------------------------
        # If the family has already started the ride everyone is "trapped".
        if self.group.parent_ref and self.group.parent_ref.has_started_ride:

            # Apply penalty: their patience time has passed and they are still waiting
            self.group.update_rating(-self.PENALTY)

            # Loop: check again after another patience cycle
            next_check = self.time + self.group.patience_limit
            simulation.schedule_event(AbandonmentEvent(next_check, self.group, self.activity))

            return # Block: do not physically abandon

        # ----------------------------------------------------------------------
        # BLOCK 2: "Trapped" teens (returned to a ride they previously abandoned)
        # ----------------------------------------------------------------------
        if self.group.type_name == "Teen" and (self.activity.activity_id in self.group.abandoned_history):

            # Penalty for waiting
            self.group.update_rating(-self.PENALTY)

            # Loop for another patience period (20 min)
            next_check = self.time + self.group.patience_limit
            simulation.schedule_event(AbandonmentEvent(next_check, self.group, self.activity))
            return

        # ------------------------------------
        # BLOCK 3: Regular abandonment attempt
        # ------------------------------------
        # This function returns the group if it successfully left the queue, or None otherwise
        abandoned_group = self.activity.queue.abandon_group(self.group, self.time, rating_penalty=self.PENALTY)

        if abandoned_group is not None:
            #  Successfully abandoned (they were in the regular queue)

            # 1.  Decision node for teens: buy an express band and return?
            if self.group.type_name == "Teen" and simulation.sampling.teen_buy_express_decision():
                if self.group.buy_express_band():
                    # "Trap": mark that they have been here so next time they will be caught in BLOCK 2
                    self.group.abandoned_history.add(self.activity.activity_id)

                    #  Return to the activity (to the express queue)
                    self.activity.add_to_queue(self.group, self.time)

                    #  Schedule a future check
                    next_check = self.time + self.group.patience_limit
                    simulation.schedule_event(AbandonmentEvent(next_check, self.group, self.activity))
                    return

            # 2. If they are teens and did not buy express
            # hey go to another ride but will come back here later
            if self.group.type_name == "Teen":
                # Save the current ride as a "ride to retry"
                self.group.retry_activity_id = self.activity.activity_id
                # for the next choice only, do not consider retry_id
                self.group.skip_retry_once = True

            # Perform the abandonment and move to the next activity
            self.activity.num_abandon += 1
            self.group.abandoned_history.add(self.activity.activity_id) # הוספה להיסטוריה (קריטי לחזרה עתידית)
            simulation.send_group_to_next_activity(abandoned_group, self.time)

        else:
            # Abandonment failed (they are in express or already on the ride)

            # Check: are they in the express queue?
            is_stuck_in_express = False
            for g, _ in self.activity.queue.express:
                if g.id == self.group.id:
                    is_stuck_in_express = True
                    break

            #  If they are teens in express apply penalty and loop
            if is_stuck_in_express and self.group.type_name == "Teen":
                self.group.update_rating(-self.PENALTY)
                next_check = self.time + self.group.patience_limit
                simulation.schedule_event(AbandonmentEvent(next_check, self.group, self.activity))

class FinishUnitEvent(Event):
    def __init__(self, time, activity, server_index):
        super().__init__(time)
        self.activity = activity
        self.server_index = server_index

    def handle(self, simulation):
        # This event releases a server, which in turn may trigger a chain of subsequent actions
        self.activity.release_finished_servers(self.server_index, self.time)

class FinishFoodPaymentEvent(Event):
    def __init__(self, time: float, stall, group):
        super().__init__(time)
        self.stall = stall      # FoodStall
        self.group = group      # VisitorGroup

    def handle(self, simulation):
        # 1) Finish service at the food stall: free the server and charge the price
        self.stall.finish_service(self.group)
        self.group.has_eaten_lunch = True
        now = simulation.current_time

        # 2) After the stall becomes available, try to start additional services
        simulation.try_start_food_services(self.stall, now)

        # 3) The group is now eating (15–35 minutes)
        eat_time = simulation.food_court.eating_duration()
        self.group.eating_until = now + eat_time
        simulation.schedule_event(FinishEatingLunchEvent(now + eat_time, self.group))
        self.group.food_payment_until = None


class FinishEatingLunchEvent(Event):
    def __init__(self, time: float, group):
        super().__init__(time)
        self.group = group

    def handle(self, simulation):
        now = simulation.current_time
        self.group.eating_until = None


        # If the group is dissatisfied with the food (probability 0.1), decrease satisfaction rating
        simulation.food_court.maybe_unsatisfied(self.group)

        # After finishing eating → proceed to the next activity
        simulation.send_group_to_next_activity(self.group, now)

class LeaveParkEvent(Event):
    def __init__(self, time, group, reason="SCHEDULED"):
        super().__init__(time)
        self.group = group
        self.reason = reason

    def handle(self, simulation):
        if self.group.finished_visits:
            return
        # case: end of simulation:
        # Do not wait: simply clear queues and exit.
        if self.reason == "EndSimulation":
            for activity in simulation.activities:
                activity.queue.remove_group_everywhere(self.group, self.time, rating_penalty=0.0)
            for stall in simulation.food_court.stalls.values():
                stall.queue.remove_group_everywhere(self.group, self.time, rating_penalty=0.0)

            self._finalize(self.time)
            return

        # 2. waiting for subgroups:
        # If the function returns a time, the group must wait.
        wait_until = self.group.collect_active_subgroups(simulation, self.time)
        if wait_until is not None:
            self.group.wants_to_leave = True
            simulation.schedule_event(
                LeaveParkEvent(wait_until + 0.1, self.group, reason="WAIT_KIDS")
            )
            return

        busy_until = self.group.busy_until()
        if busy_until is not None:
            simulation.schedule_event(
                LeaveParkEvent(busy_until, self.group, reason="WAIT_ACTIVITY_END")
            )
            return

       # 3. Perform regular departure
        for activity in simulation.activities:
            activity.queue.remove_group_everywhere(self.group, self.time, rating_penalty=0.0)
        for stall in simulation.food_court.stalls.values():
            stall.queue.remove_group_everywhere(self.group, self.time, rating_penalty=0.0)

        self._finalize(self.time)

    #charging for pictures according to rating
    def _finalize(self, now):
        rating = self.group.current_rating
        photo_revenue = 0
        if 6.0 <= rating < 7.5: photo_revenue = 20
        elif 7.5 <= rating < 8.5: photo_revenue = 100
        elif rating >= 8.5: photo_revenue = 120

        if photo_revenue > 0:
            self.group.record_spending(photo_revenue)

        self.group.finished_visits = True
        self.group.stats['total_time'] = now - self.group.arrival_time

class EndSimulationEvent(Event):
    def __init__(self, time: float):
        super().__init__(time)

    def handle(self, simulation):
        now = self.time
        simulation.current_time = now
        simulation.ended = True
     # 1) stop all future events
        simulation.event_heap.clear()
        # unit family before calculations
       # 1. who is in the queue to the entrance queue?
        entrance = simulation.entrance_station
        groups_in_entrance_queue = set()
        for g, _ in entrance.queue.regular:
            groups_in_entrance_queue.add(g.id)
        for g, _ in entrance.queue.express:
            groups_in_entrance_queue.add(g.id)

        # 2. handle visitors inside the park
        for group in simulation.visitors:
            if group.finished_visits: continue
            if group.id in groups_in_entrance_queue: continue # dont thouce the visitors who didnt enter the park yet

            # force unit and get future exit time
            real_exit_time = group.force_aggregate_statistics(now)

            # detachment
            LeaveParkEvent(now, group, reason="EndSimulation").handle(simulation)

            # calculations and closure
            rating = group.current_rating
            photo_revenue = 0
            if 6.0 <= rating < 7.5: photo_revenue = 20
            elif 7.5 <= rating < 8.5: photo_revenue = 100
            elif rating >= 8.5: photo_revenue = 120
            if photo_revenue > 0:
                group.record_spending(photo_revenue)
            group.finished_visits = True
            group.stats['total_time'] = real_exit_time - group.arrival_time

        # 3. handle visitors in entrance
        entrance_busy_before = entrance.busy
        evacuated_from_entrance = entrance.queue.evacuate_all(now)
        simulation.evacuated_total = len(evacuated_from_entrance)
        entrance.busy = 0

        for g in evacuated_from_entrance:
            g.finished_visits = True
            g.stats['total_time'] = 0 # didnt enter

  # 4. clean the activities and collect data

        # food stalls
        stalls_snapshot = {}
        for name, stall in simulation.food_court.stalls.items():
            stall.queue.evacuate_all(now)
            stall.server_busy = False

            served = stall.num_served
            avg_service = (stall.total_service_time / served) if served > 0 else 0.0

            stalls_snapshot[name] = {
                "served": served,
                "revenue": stall.revenue,
                "avg_service_time": avg_service,
                "queue_avg_wait": stall.queue.average_wait_time_minutes(),
                "queue_empty_time": stall.queue.empty_time_minutes,
                "server_busy_before_close": False,
            }

        # activities
        activities_snapshot = {}
        for a in simulation.activities:
            a.queue.evacuate_all(now)
            for i in range(a.max_units_parallel):
                a.is_busy[i] = False
                a.server_groups[i] = None
                a.server_num_people[i] = 0
                a.inflight[i].clear()

            units = a.total_units_started
            avg_unit_time = (a.total_service_time / units) if units > 0 else 0.0

            activities_snapshot[a.activity_id] = {
                "name": a.name,
                "units_started": units,
                "people_served": a.total_people_served,
                "total_service_time": a.total_service_time,
                "avg_unit_time": avg_unit_time,
                "num_abandon": a.num_abandon,
                "queue_avg_wait": a.queue.average_wait_time_minutes(),
                "queue_empty_time": a.queue.empty_time_minutes,
            }

        # 5. overall summary
        # filter: only groups that actually entered the park
        groups_who_entered = [g for g in simulation.visitors if g.stats['total_time'] > 0]

        # New variables for accurate calculations
        count_entered = len(groups_who_entered)
        people_entered = sum(g.size for g in groups_who_entered)

        # sum just who enterd
        total_money_entered = sum(g.stats.get('money_spent', 0.0) for g in groups_who_entered)
        total_wait_entered  = sum(g.stats.get('total_wait_time', 0.0) for g in groups_who_entered)

        #calculate average
        avg_money_per_group = (total_money_entered / count_entered) if count_entered > 0 else 0.0
        avg_wait_per_group  = (total_wait_entered / count_entered) if count_entered > 0 else 0.0
        avg_money_per_person = (total_money_entered / people_entered) if people_entered > 0 else 0.0

        total_money = total_money_entered # only group that entered the park paid
        total_food_revenue = sum(stall.revenue for stall in simulation.food_court.stalls.values())

        entrance_avg_wait = entrance.queue.average_wait_time_minutes()
        entrance_empty_time = entrance.queue.empty_time_minutes
        # Number of people in the entrance queue before closing is exactly the number of evacuated groups
        entrance_queue_people_before = simulation.evacuated_total

        simulation.final_report = {
            "end_time": "19:00",
            "evacuated_total": simulation.evacuated_total,
            "groups_total": len(simulation.visitors), # total groups that arrives
            "groups_entered": count_entered,          # total group that entered
            "people_total": sum(g.size for g in simulation.visitors),

            "money_total": total_money,
            "wait_total": total_wait_entered,

            "avg_money_per_group": avg_money_per_group,
            "avg_money_per_person": avg_money_per_person,
            "avg_wait_per_group": avg_wait_per_group,

            "entrance": {
                "ticket_revenue": entrance.total_ticket_revenue,
                "express_revenue": entrance.total_express_revenue,
                "queue_avg_wait": entrance_avg_wait,
                "queue_empty_time": entrance_empty_time,
                "busy_servers_before_close": entrance_busy_before,
            },
            "food_total_revenue": total_food_revenue,
            "food_stalls": stalls_snapshot,
            "activities": activities_snapshot,
        }

class LunchBreakEvent(Event):  # Lunch break for instructors (snorkeling)

    def __init__(self, time: float, activity: "Activity", is_start: bool):
        super().__init__(time)
        self.activity = activity
        self.is_start = is_start # Indicates whether this is the start or the end of the lunch break (True = start)

    def handle(self, simulation):
        if self.is_start: # Start of lunch break
            for i in range(self.activity.max_units_parallel): # Iterate over all instructors
                self.activity.instructor_on_lunch[i] = True #Mark instructors as on lunch break


            # Schedule the end of the lunch break 60 minutes later
            simulation.schedule_event(LunchBreakEvent(self.time + 60, self.activity, is_start=False))

        else:
            # End of lunch break
            for i in range(self.activity.max_units_parallel): # Iterate over instructors
                self.activity.instructor_on_lunch[i] = False # Mark instructors as no longer on break

            # new sessions can be started if there is a queue
            self.activity.try_start_units(self.time)





class Simulation:
    def __init__(self, open_time, close_time, sampling, config=None):
        #Times in minutes
        self.open_time = open_time
        self.close_time = close_time
        self.current_time = open_time
        self.sampling = sampling

        if config is None: config = {}
        self.has_website = config.get('has_website', False)
        self.wave_pool_cap = config.get('wave_pool_cap', 80)
        self.initial_rating = config.get('initial_rating', 10.0)
        kitchen_improved = config.get('kitchen_improved', False)

        self.sampling.set_kitchen_quality(kitchen_improved)
        # -----------------------------------------------
        self.entrance_station = EntranceStation(self.sampling, num_servers=3, has_website=self.has_website)
        self.food_court = FoodCourt(self.sampling)
        self.snorkel_lunch_start = 13*60
        self.snorkel_lunch_end = 14*60

        self.visitors = []  ## list of groups of visitors
        self.total_groups = 0 ## number of groups
        self.next_group_id = 1
        self.activities = [] ## list of activities

        #  heap- event log
        self.event_heap = []
        self._event_seq = 0 #Internal counter to break ties between events at the same time
        # self.event_log = []   # all events that actually executed


        ##Initialize initial events
        # push the first three arrival events into the heap
        lam_family = 40.0 / 60.0
        lam_teens  = 500.0 / 360.0 # teens arrive from 10 to 16
        lam_single = 10.0 / 15.0

        t1_family = self.open_time + self.sampling.exponential(lam_family)
        t1_teens  = 10*60 + self.sampling.exponential(lam_teens) #arrivals start from 10:00
        t1_single = self.open_time + self.sampling.exponential(lam_single)

        if t1_family <= 12*60:
            self.schedule_event(FamilyArrivalEvent(t1_family))

        if t1_teens <= 16*60:
            self.schedule_event(TeensArrivalEvent(t1_teens))

        if t1_single <= (self.close_time - 30):
            self.schedule_event(SingleArrivalEvent(t1_single))

        ## creat end simulation event
        self.ended = False
        self.evacuated_total = 0
        self.final_report = None

        self.schedule_event(EndSimulationEvent(self.close_time))  # close_time = 19*60


        self.init_activities()

    # ----------push event into the heap------------
    def schedule_event(self, event: "Event"):
      self._event_seq += 1
      heapq.heappush(self.event_heap, (event.time, self._event_seq, event))


    #------creat ID-----------
    def new_id(self):
        self.next_group_id += 1
        return (self.next_group_id - 1)

    # -------------- creat visitors----------------------
    def create_family(self):
        # 1) how many children?
        num_children = self.sampling.get_num_children_family()

        # 2) sample an age for each child
        children_ages = []
        for _ in range(num_children):
            child_age = int(self.sampling.get_child_age())
            children_ages.append(child_age)

        # 3) add parent in the age of 30
        ages_list = [30, 30] + children_ages

        # 4) creat real family object
        group_id = self.new_id()
        size = len(ages_list)
        return Family(group_id, size, self.current_time, ages_list, initial_rating=self.initial_rating)
    def create_teens(self):
        group_size = self.sampling.get_teen_group_size()  # דגימה לגודל קבוצה
        group_id = self.new_id()
        return TeenGroup(group_id, group_size, self.current_time, initial_rating=self.initial_rating)

    def create_single(self):
        group_id = self.new_id()
        return Single(group_id, self.current_time, initial_rating=self.initial_rating)

    # ---------------- arrival events management---------------------

    def try_start_entrance_services(self, now_minutes):
      while self.entrance_station.can_start():
          result = self.entrance_station.start_service(now_minutes)
          if result is None:
              break

          group, finish_time = result

          # Schedule a cashier finish event as an object
          self.schedule_event(FinishEntranceEvent(finish_time, group))

    def handle_family_arrival(self):
      group = self.create_family()
      self.visitors.append(group)
      self.total_groups += 1
      self.entrance_station.enqueue(group, self.current_time)
      self.try_start_entrance_services(self.current_time)
      leave_hour = self.sampling.get_family_leave_time()
      leave_time = leave_hour * 60  # convers to minutes

      event = LeaveParkEvent(leave_time, group, reason="SCHEDULED")
      self.schedule_event(event)

    def handle_teens_arrival(self):
      group = self.create_teens()
      self.visitors.append(group)
      self.total_groups += 1
      self.entrance_station.enqueue(group, self.current_time)
      self.try_start_entrance_services(self.current_time)

    def handle_single_arrival(self):
      group = self.create_single()
      self.visitors.append(group)
      self.total_groups += 1
      self.entrance_station.enqueue(group, self.current_time)
      self.try_start_entrance_services(self.current_time)

    #------------lunch event management--------------

    def try_start_food_services(self, stall, now_minutes):
        while stall.can_start_service():
            result = stall.start_service(now_minutes)
            if result is None:
                break

            group, finish_time = result
            group.food_payment_until = finish_time

            self.schedule_event(FinishFoodPaymentEvent(finish_time, stall, group))

    def send_group_to_food(self, group, now_minutes):
        stall_name = self.food_court.choose_stall_name()   # Burger / Pizza / Salad
        stall = self.food_court.stalls[stall_name]

        stall.add_to_queue(group, now_minutes)

        # If the stall is free — start immediately and schedule FinishFoodPaymentEvent
        self.try_start_food_services(stall, now_minutes)

        return True


   #---------------------navigation related functions------------------
    #returns the activity with the shortest queue
    def _choose_shortest_queue(self, candidates, group):
      best = None
      best_q = float("inf") #positive infinity

      for a in candidates: #loop over all activities
        if group.has_express_band:
          # counts only express queue
          q = 0 #people in queue counter
          for g, _ in a.queue.express:
            q += g.size

        else:
          # counts only regular queue
          q = 0
          for g, _ in a.queue.regular:
            q += g.size

        if q < best_q:
          best_q = q
          best = a #best activity

      return best

    def _choose_next_activity(self, group):
      allowed = self._allowed_activities(group)
      if not allowed:
        return None

      # --- families ---
      if group.type_name == "Family":
        #Has the family already completed all age-appropriate activities?
        for i in allowed:
          if i.activity_id == 1 or i.activity_id == 3:
            return i
        if group.parent_ref == None:
          groups = group.try_split(self.sampling)
          for g in groups:
            self.send_group_to_next_activity(g, self.current_time)
          return None
        else:
          parent = group.parent_ref
          if parent.total_siblings == 0:
            groups = group.try_split(self.sampling)
            for g in groups:
              self.send_group_to_next_activity(g, self.current_time)
            return None
        return self._choose_shortest_queue(allowed, group)

      # --- teens---
      if group.type_name == "Teen":
        # 1. retry mechanism check
            #We enter here only if there is a ride to return to, and "skip" is off
        if (group.retry_activity_id is not None) and (not group.skip_retry_once):

          target_activity = None
          for act in self.activities:
            if act.activity_id == group.retry_activity_id:
              target_activity = act
              break

          #reset memory and return to the ride
          if target_activity:
            group.retry_activity_id = None
            return target_activity

        # 2.  If we got here, it means we are in a regular cho

        #  If the flag was up — lower it now (so next time they WILL retry)
        if group.skip_retry_once:
          group.skip_retry_once = False

        # regular choice
        adrenaline_ok = [a for a in allowed if a.adrenaline >= 3]

        # Filter: if we have a ride waiting for Retry, we must not pick it now
        # (We want to go to a *different* ride)
        if group.retry_activity_id is not None:
          adrenaline_ok = [a for a in adrenaline_ok if a.activity_id != group.retry_activity_id]

        if not adrenaline_ok:
          return None

        idx = self.sampling.uniform_discrete(0, len(adrenaline_ok) - 1)
        return adrenaline_ok[idx]

      # --- singles ---
      if group.type_name == "Single":
          # first- 12 or higher ages activities
          prefer_12 = [a for a in allowed if a.min_age >= 12]
          if prefer_12:
              return self._choose_shortest_queue(prefer_12, group)

          ##then all of the others activities
          allowed = [a for a in allowed if a.activity_id != 6]
          return self._choose_shortest_queue(allowed, group)

          return None

      # default
      return self._choose_shortest_queue(allowed, group)

    #Receives a group and returns a list of activities it is allowed to go to
    # based on age and what it has already visited
    def _allowed_activities(self, group):
      allowed = []
      for a in self.activities:
        if group.has_visited(a.activity_id):
          continue
        if a.activity_id == 6: ##toddlers pool
            # Only if there is at least one child age 4 or under (in a family)
            if group.type_name == "Family" and any(age <= 4 for age in group.ages):
              allowed.append(a)
            continue
        if not group.can_ride(a):
          continue
        allowed.append(a)
      return allowed


    #Actually sends a group to the next activity
    def send_group_to_next_activity(self, group, now_minutes):
        # Guard: if the leaving family is currently on a ride
        if group.finished_visits:
             return False
        activity = self._choose_next_activity(group)
        if activity is None:
          self.schedule_event(LeaveParkEvent(now_minutes + 1, group, reason = "FINNISHED_ALL"))
          return False
        return activity.add_to_queue(group, now_minutes)


    #-----------Initialize activities-----------
    def init_activities(self):

      self.activities = []

      # 1) Tube River
      river = Activity(
        simulation=self,
        activity_id=1,
        name="Tube River",
        min_age=0,
        adrenaline=1,
        unit_capacity=2,
        max_units_parallel=60,
        duration_sampler= lambda: self.sampling.uniform(20,30)
      )
      self.activities.append(river)

      # 2) Single Slides (2 slides)
      single_slides = Activity(
        simulation=self,
        activity_id=2,
        name="Single Slides",
        min_age=14,
        adrenaline=5,
        unit_capacity=1,
        max_units_parallel=2,           # 2 slides
        duration_sampler=lambda: 3.0    # exactly 3 minutes
      )
      # safety gap: 30 seconds between entries PER SLIDE (per server)
      single_slides.min_entry_gap = 0.5  # minutes
      self.activities.append(single_slides)

      # 3) Big Tube Slide (exactly 8 per tube, one tube at a time)
      big_tube = Activity(
        simulation=self,
        activity_id=3,
        name="Big Tube Slide",
        min_age=0,
        adrenaline=2,
        unit_capacity=8,
        max_units_parallel=1,
        duration_sampler=lambda: self.sampling.normal(4.801, 1.823)
      )
      self.activities.append(big_tube)

      # 4) Small Tube Slide (exactly 3 per tube, one tube at a time, age 12+)
      small_tube = Activity(
        simulation=self,
        activity_id=4,
        name="Small Tube Slide",
        min_age=12,
        adrenaline=4,
        unit_capacity=3,
        max_units_parallel=1,
          duration_sampler=lambda: self.sampling.exponential(1/2.107)
      )
      self.activities.append(small_tube)

      # 5) Wave Pool (capacity 80 concurrent)
      wave_pool = Activity(
        simulation=self,
        activity_id=5,
        name="Wave Pool",
        min_age=12,
        adrenaline=3,
        unit_capacity=self.wave_pool_cap,
        max_units_parallel=1,
        duration_sampler= lambda: self.sampling.get_wave_pool_duration()
      )
      self.activities.append(wave_pool)

      # 6) Toddler Pool (capacity 30 children concurrent)
      toddler_pool = Activity(
        simulation=self,
        activity_id=6,
        name="Toddler Pool",
        min_age=0,         # age rule handled in your special toddler logic
        adrenaline=1,
        unit_capacity=30,  # children capacity
        max_units_parallel=1,
        duration_sampler=lambda: self.sampling.Toddler_pool()
      )
      self.activities.append(toddler_pool)

      # 7) Snorkel Tour (2 instructors)
      snorkel = Activity(
        simulation=self,
        activity_id=7,
        name="Snorkel Tour",
        min_age=6,
        adrenaline=3,
        unit_capacity=30,             # up to 30 per tour
        max_units_parallel=2,
        duration_sampler=lambda: self.sampling.normal(30, 10)
      )
      self.schedule_event(LunchBreakEvent(13*60, snorkel, True))
      self.activities.append(snorkel)

    #def print_event_log(self, limit=None):
     #   print("\n--- EVENT LOG (EXECUTED EVENTS) ---")
      #  for i, (t, name) in enumerate(self.event_log):
       #     if limit and i >= limit:
        #        print("...")
         #       break
          #  print(f"{i:4d} | time={t:7.2f} | {name}")


    # ------------- run -------------------------
    def run(self):
      while self.event_heap:
        event_time, seq, event = heapq.heappop(self.event_heap)
       # self.event_log.append((event_time, event.__class__.__name__))

        if event_time > self.close_time:
            break

        self.current_time = event_time
        event.handle(self)

        if self.ended:
            break

      return self.final_report

import numpy as np
import scipy.stats as st
import math
import heapq
import matplotlib.pyplot as plt


#--------------
# VISUALIZATION
#--------------

def plot_simulation_results(final_results, alpha=0.1):

    #Function for creating 3 summary charts (profit, unserved customers, and average waiting time)

    plt.style.use('seaborn-v0_8-whitegrid')
    scenario_names = list(final_results.keys()) # ['Base Case', 'Combination A', 'Combination B']

    # Compute sample size n and critical t-value
    n = len(final_results[scenario_names[0]]['revenue'])
    t_crit = st.t.ppf(1 - alpha/2, n - 1)

    # Metrics configuration: (dictionary key, Y-axis label, plot title, bar colors)
    metrics_config = [
        ('revenue', 'Total Profit (ILS)', '1. Total Daily Profit', ['gray', 'green', 'green']),
        ('unserved', 'Unserved Customers', '2. Unserved Customers', ['gray', 'red', 'red']),
        ('avg_wait', 'Avg Wait Time (Minutes)', '3. Average Wait Time', ['gray', 'orange', 'blue'])
    ]

    # Create a figure with 3 subplots in a single row
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (metric_key, ylabel, title, bar_colors) in enumerate(metrics_config):
        ax = axes[i]

        means = []
        ci_errs = [] # Confidence interval half-widths

        # Compute statistics for each scenario
        for name in scenario_names:
            data = np.array(final_results[name][metric_key])
            mean_val = np.mean(data)
            std_val = np.std(data, ddof=1)
            se = std_val / np.sqrt(n)
            h = t_crit * se # Confidence interval half-width

            means.append(mean_val)
            ci_errs.append(h)

        #  Draw bar chart
        bars = ax.bar(scenario_names, means, yerr=ci_errs, capsize=10,
                      color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)

        # Plot formatting
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='x', rotation=15)

        #  Add numeric labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            if height > 1000:
                label = f'{height:,.0f}'
            else:
                label = f'{height:.1f}'

            ax.text(bar.get_x() + bar.get_width()/2., height + (max(ci_errs)*1.1),
                    label,
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

    # Main title for the entire figure
    plt.suptitle(f'Simulation Results Comparison (n={n} runs, {int((1-alpha)*100)}% Confidence)', fontsize=16, y=0.98)

    plt.tight_layout()
    #  Save figure to file
    plt.savefig("final_presentation_charts.png", dpi=300)
    print("\nGraph saved as 'final_presentation_charts.png'")
    plt.show()


#---------------------
# MAIN EXECUTION BLOCK
#---------------------

if __name__ == "__main__":

    print("\n=== Sample Size Determination & Final Analysis ===")

    # 1. Define constants
    OPEN_TIME = 9 * 60
    CLOSE_TIME = 19 * 60
    N0 = 30
    GAMMA_TARGET = 0.1
    ALPHA_TOTAL = 0.1

    # 2. Define simulation scenarios
    scenarios = {
        "Base Case": {
            'config': {
                'has_website': False,
                'kitchen_improved': False,
                'wave_pool_cap': 80,      # Default value
                'initial_rating': 10.0     # Regular initial rating
            }
        },
        "Combination A (Web + Kitchen + Visitor Benefit)": {
            # Cost: 100K + 100K + 50K = 250K
            'config': {
                'has_website': True,
                'kitchen_improved': True,
                'wave_pool_cap': 80,       # No change
                'initial_rating': 11.0
            }
        },
        "Combination B (Web + Wave + Visitor Benefit)": {
            # Cost: 100K (website) + 70K (wave pool) + 50K (benefit) = 220K
            'config': {
                'has_website': True,
                'kitchen_improved': False,
                'wave_pool_cap': 120,        # Increased wave pool capacity
                'initial_rating': 11.0
            }
        }
    }

    metrics = ['revenue', 'unserved', 'avg_wait']

    #Bonferroni adjusted alpha and critical t-value
    alpha_bonf_n = ALPHA_TOTAL / (len(metrics) * len(scenarios))
    t_crit_n0 = st.t.ppf(1 - alpha_bonf_n/2, N0 - 1)

    max_n_needed = N0
    worst_metric_info = "None"

    # Initial runs (n=30) and table output
    print("\n--- Phase 1: Initial Runs (n=30) ---")

    # Loop that prints a results table for each scenario
    for name, data in scenarios.items():
        print(f"\n>> Simulation Results for: {name} (n={N0})")
        print(f"{'Metric':<20} | {'Mean':<10} | {'StdDev':<10} | {'Half-Width':<12} | {'Rel. Prec':<10} | {'Status'}")
        print("-" * 90)

        results = {'revenue': [], 'unserved': [], 'avg_wait': []}

        # Run 30 simulation replications
        for i in range(30):
            current_seed = 42 + i
            sampler = SamplingAlgorithms(seed=current_seed)
            sim = Simulation(open_time=OPEN_TIME, close_time=CLOSE_TIME, sampling=sampler, config=data['config'])
            report = sim.run()

            results['revenue'].append(report['money_total'])
            results['unserved'].append(report['evacuated_total'])
            val = report['wait_total'] / report['groups_total'] if report['groups_total'] > 0 else 0
            results['avg_wait'].append(val)

        # Compute statistics and print results
        for metric_name, values in results.items():
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)

            # Confidence interval half-width
            h0 = t_crit_n0 * (std_val / np.sqrt(N0))

            # Relative precision (for display)
            rel_prec_display = (h0 / abs(mean_val)) if mean_val != 0 else 0

            #  Adjusted precision target
            if mean_val != 0:
                target_hw = (GAMMA_TARGET * abs(mean_val)) / (1 + GAMMA_TARGET)
                is_ok = h0 <= target_hw
            else:
                is_ok = True

            status = "OK" if is_ok else "FAIL"

            print(f"{metric_name:<20} | {mean_val:<10.1f} | {std_val:<10.1f} | {h0:<12.2f} | {rel_prec_display:<10.4f} | {status}")

            #  Compute required sample size if precision target is not met
            if not is_ok:
                # n = n0 * (h0 / target_hw)^2
                ratio = h0 / target_hw
                n_calc = int(np.ceil(N0 * (ratio**2)))
                if n_calc > max_n_needed:
                    max_n_needed = n_calc
                    worst_metric_info = f"{name}-{metric_name}"

    print("\n" + "="*60)
    print(f"SAMPLE SIZE DECISION: Worst case ({worst_metric_info}) requires N = {max_n_needed}")
    print("="*60)


    # Final simulation and hypothesis testing

    FINAL_N = max_n_needed
    print(f"\n>> Running Final Simulation with N={FINAL_N} (CRN)...")

    final_results = {s: {m: [] for m in metrics} for s in scenarios}

    # Final simulation runs
    for i in range(FINAL_N):
        current_seed = 42 + i
        for name, data in scenarios.items():
            sampler = SamplingAlgorithms(seed=current_seed)
            sim = Simulation(open_time=OPEN_TIME, close_time=CLOSE_TIME, sampling=sampler, config=data['config'])
            report = sim.run()

            final_results[name]['revenue'].append(report['money_total'])
            final_results[name]['unserved'].append(report['evacuated_total'])
            val = report['wait_total'] / report['groups_total'] if report['groups_total'] > 0 else 0
            final_results[name]['avg_wait'].append(val)

    # Paired t-tests:
    print("\n=== HYPOTHESIS TESTING (Paired t-test) ===")


    comparisons = [
        ("Combination A (Web + Kitchen + Visitor Benefit)", "Base Case"),
        ("Combination B (Web + Wave + Visitor Benefit)", "Base Case"),
        ("Combination A (Web + Kitchen + Visitor Benefit)", "Combination B (Web + Wave + Visitor Benefit)")
    ]

    # Final critical values
    alpha_comp = ALPHA_TOTAL / (len(comparisons) * len(metrics))
    df_final = FINAL_N - 1
    t_crit_final = st.t.ppf(1 - alpha_comp/2, df_final)

    print(f"Final N: {FINAL_N}, df: {df_final}, Critical T: {t_crit_final:.4f}")
    print("-" * 105)
    print(f"{'Pair':<30} | {'Metric':<12} | {'Mean Diff':<10} | {'t-Stat':<10} | {'Reject H0?':<10} | {'CI (Diff)'}")
    print("-" * 105)

    for (n1, n2) in comparisons:
        for m in metrics:
            d1 = np.array(final_results[n1][m])
            d2 = np.array(final_results[n2][m])
            diffs = d1 - d2

            mean_d = np.mean(diffs)
            std_d = np.std(diffs, ddof=1)
            se_d = std_d / np.sqrt(FINAL_N)

            # t-statistic
            t_stat = mean_d / se_d if se_d != 0 else 0

            #  Reject H0?
            reject = "YES" if abs(t_stat) > t_crit_final else "NO"

            # Confidence interval for the difference
            hw_d = t_crit_final * se_d
            ci = f"[{mean_d - hw_d:.1f}, {mean_d + hw_d:.1f}]"

            print(f"{n1} - {n2:<9} | {m:<12} | {mean_d:<10.1f} | {t_stat:<10.2f} | {reject:<10} | {ci}")
    print("="*105)

    # Generate final presentation charts
    print("\nGenerating Presentation Graphs...")
    plot_simulation_results(final_results)
