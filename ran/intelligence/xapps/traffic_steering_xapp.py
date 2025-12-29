"""
Traffic Steering xApp for Near-RT RIC

Implements intelligent traffic steering using AI/ML models for:
- Load balancing across cells
- Inter-frequency handover decisions
- Dual connectivity optimization
- Traffic offloading to small cells
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any

from ..ric.near_rt_ric import xApp

logger = logging.getLogger(__name__)


class TrafficSteeringxApp(xApp):
    """
    Traffic Steering xApp
    
    Makes intelligent decisions to steer traffic between cells based on:
    - Cell load
    - Channel quality
    - User mobility
    - QoS requirements
    """
    
    def __init__(self, xapp_id: str = "traffic-steering-001"):
        """Initialize Traffic Steering xApp"""
        super().__init__(xapp_id, "Traffic Steering xApp")
        
        # Configuration
        self.config = {
            'load_threshold': 0.8,  # 80% load threshold
            'steering_margin_db': 3.0,  # 3dB margin for handover
            'min_time_between_decisions_ms': 100,  # Minimum 100ms between decisions
            'max_handovers_per_interval': 10
        }
        
        # State tracking
        self.cell_states: Dict[str, Dict] = {}
        self.user_states: Dict[str, Dict] = {}
        self.steering_decisions: List[Dict] = []
        self.last_decision_time = 0
        
        # Performance tracking
        self.successful_steerings = 0
        self.failed_steerings = 0
        self.total_decisions = 0
    
    def on_start(self):
        """Callback when xApp starts"""
        logger.info("Traffic Steering xApp started")
        self.update_metrics({
            'status': 'active',
            'steering_policy': 'load_aware_quality_based'
        })
    
    def on_stop(self):
        """Callback when xApp stops"""
        logger.info("Traffic Steering xApp stopped")
        logger.info(f"Total steering decisions: {self.total_decisions}")
        logger.info(f"Success rate: {self.successful_steerings / max(self.total_decisions, 1) * 100:.1f}%")
    
    def handle_indication(self, indication_data: Dict[str, Any]):
        """
        Handle indication from RAN
        
        Args:
            indication_data: Indication data including cell load, measurements
        """
        # Update cell states
        if 'cell_measurements' in indication_data:
            self._update_cell_states(indication_data['cell_measurements'])
        
        # Update user states
        if 'user_measurements' in indication_data:
            self._update_user_states(indication_data['user_measurements'])
    
    def make_control_decision(self, ran_state: Dict[str, Any]) -> Optional[Dict]:
        """
        Make traffic steering control decision
        
        Args:
            ran_state: Current RAN state
            
        Returns:
            dict: Control action for traffic steering, or None
        """
        import time
        current_time = time.time() * 1000  # ms
        
        # Rate limit decisions
        if current_time - self.last_decision_time < self.config['min_time_between_decisions_ms']:
            return None
        
        # Identify overloaded cells
        overloaded_cells = self._identify_overloaded_cells()
        
        if not overloaded_cells:
            return None
        
        # Generate steering decisions
        steering_actions = []
        
        for cell_id in overloaded_cells:
            actions = self._generate_steering_actions(cell_id, ran_state)
            steering_actions.extend(actions)
        
        if not steering_actions:
            return None
        
        # Limit number of simultaneous handovers
        steering_actions = steering_actions[:self.config['max_handovers_per_interval']]
        
        self.last_decision_time = current_time
        self.total_decisions += len(steering_actions)
        
        # Create control action
        control_action = {
            'node_id': ran_state.get('node_id', 'unknown'),
            'action': 'traffic_steering',
            'ran_function_id': 'traffic_steering',
            'parameters': {
                'steering_actions': steering_actions
            }
        }
        
        # Track decision
        self.steering_decisions.append({
            'timestamp': current_time,
            'actions': steering_actions
        })
        
        self.update_metrics({
            'total_decisions': self.total_decisions,
            'pending_actions': len(steering_actions)
        })
        
        return control_action
    
    def _update_cell_states(self, cell_measurements: Dict):
        """Update internal cell state tracking"""
        for cell_id, measurements in cell_measurements.items():
            if cell_id not in self.cell_states:
                self.cell_states[cell_id] = {}
            
            self.cell_states[cell_id].update({
                'load': measurements.get('load', 0),
                'active_users': measurements.get('active_users', 0),
                'avg_throughput': measurements.get('avg_throughput', 0),
                'prb_utilization': measurements.get('prb_utilization', 0),
                'timestamp': measurements.get('timestamp', 0)
            })
    
    def _update_user_states(self, user_measurements: Dict):
        """Update internal user state tracking"""
        for user_id, measurements in user_measurements.items():
            if user_id not in self.user_states:
                self.user_states[user_id] = {}
            
            self.user_states[user_id].update({
                'serving_cell': measurements.get('serving_cell'),
                'rsrp': measurements.get('rsrp', -140),
                'rsrq': measurements.get('rsrq', -20),
                'neighbor_cells': measurements.get('neighbor_cells', []),
                'throughput': measurements.get('throughput', 0),
                'qos_class': measurements.get('qos_class', 9)
            })
    
    def _identify_overloaded_cells(self) -> List[str]:
        """
        Identify cells exceeding load threshold
        
        Returns:
            list: Cell IDs of overloaded cells
        """
        overloaded = []
        
        for cell_id, state in self.cell_states.items():
            load = state.get('load', 0)
            if load > self.config['load_threshold']:
                overloaded.append(cell_id)
                logger.info(f"Cell {cell_id} overloaded: {load * 100:.1f}%")
        
        return overloaded
    
    def _generate_steering_actions(self, source_cell_id: str,
                                   ran_state: Dict) -> List[Dict]:
        """
        Generate steering actions for an overloaded cell
        
        Args:
            source_cell_id: ID of overloaded cell
            ran_state: Current RAN state
            
        Returns:
            list: List of steering actions
        """
        actions = []
        
        # Get users in the overloaded cell
        users_in_cell = [
            user_id for user_id, state in self.user_states.items()
            if state.get('serving_cell') == source_cell_id
        ]
        
        # For each user, check if steering is beneficial
        for user_id in users_in_cell:
            user_state = self.user_states[user_id]
            
            # Find best target cell
            target_cell = self._find_best_target_cell(user_id, user_state)
            
            if target_cell and target_cell != source_cell_id:
                # Check steering conditions
                if self._should_steer_user(user_id, user_state, target_cell):
                    actions.append({
                        'type': 'handover',
                        'user_id': user_id,
                        'source_cell': source_cell_id,
                        'target_cell': target_cell,
                        'reason': 'load_balancing'
                    })
        
        return actions
    
    def _find_best_target_cell(self, user_id: str, user_state: Dict) -> Optional[str]:
        """
        Find best target cell for user
        
        Args:
            user_id: User identifier
            user_state: User state information
            
        Returns:
            str: Best target cell ID, or None
        """
        neighbor_cells = user_state.get('neighbor_cells', [])
        current_rsrp = user_state.get('rsrp', -140)
        
        best_cell = None
        best_score = -float('inf')
        
        for neighbor in neighbor_cells:
            cell_id = neighbor.get('cell_id')
            rsrp = neighbor.get('rsrp', -140)
            
            # Skip if cell is unknown
            if cell_id not in self.cell_states:
                continue
            
            cell_load = self.cell_states[cell_id].get('load', 1.0)
            
            # Compute steering score (higher is better)
            # Consider both signal strength and load
            signal_benefit = rsrp - current_rsrp
            load_benefit = 1.0 - cell_load
            
            score = signal_benefit + 20 * load_benefit  # Weight load heavily
            
            if score > best_score:
                best_score = score
                best_cell = cell_id
        
        return best_cell
    
    def _should_steer_user(self, user_id: str, user_state: Dict,
                          target_cell: str) -> bool:
        """
        Determine if user should be steered to target cell
        
        Args:
            user_id: User identifier
            user_state: User state
            target_cell: Target cell ID
            
        Returns:
            bool: True if steering recommended
        """
        # Check signal strength margin
        current_rsrp = user_state.get('rsrp', -140)
        
        # Find target cell RSRP from neighbor list
        target_rsrp = -140
        for neighbor in user_state.get('neighbor_cells', []):
            if neighbor.get('cell_id') == target_cell:
                target_rsrp = neighbor.get('rsrp', -140)
                break
        
        # Require minimum signal margin to avoid ping-pong
        if target_rsrp < current_rsrp + self.config['steering_margin_db']:
            return False
        
        # Check target cell load
        if target_cell in self.cell_states:
            target_load = self.cell_states[target_cell].get('load', 1.0)
            if target_load > self.config['load_threshold']:
                return False
        
        # Check QoS class - prioritize best-effort traffic for steering
        qos_class = user_state.get('qos_class', 9)
        if qos_class < 5:  # Don't steer high-priority traffic
            return False
        
        return True
    
    def get_statistics(self) -> Dict:
        """Get xApp statistics"""
        return {
            'total_decisions': self.total_decisions,
            'successful_steerings': self.successful_steerings,
            'failed_steerings': self.failed_steerings,
            'success_rate': self.successful_steerings / max(self.total_decisions, 1),
            'tracked_cells': len(self.cell_states),
            'tracked_users': len(self.user_states),
            'recent_decisions': len(self.steering_decisions[-10:])
        }
