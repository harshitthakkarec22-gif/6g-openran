"""
Near-Real-Time RAN Intelligent Controller (Near-RT RIC)

Implements the Near-RT RIC component for 6G OPENRAN architecture.
Operates with sub-10ms control loop latency for real-time RAN optimization.

Based on O-RAN specifications and latest 6G research.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from queue import Queue
import numpy as np

from .e2_interface import E2Interface, E2Message, E2MessageType, E2NodeConfig

logger = logging.getLogger(__name__)


@dataclass
class RICConfig:
    """Configuration for Near-RT RIC"""
    ric_id: str
    control_loop_interval_ms: int = 10  # 10ms for near-real-time
    max_xapps: int = 50
    e2_port: int = 36421
    a1_endpoint: str = "http://non-rt-ric:8080"


class xApp:
    """
    Base class for xApps (RIC applications)
    
    xApps implement specific RAN control logic using AI/ML models
    """
    
    def __init__(self, xapp_id: str, xapp_name: str):
        """
        Initialize xApp
        
        Args:
            xapp_id: Unique xApp identifier
            xapp_name: Human-readable xApp name
        """
        self.xapp_id = xapp_id
        self.xapp_name = xapp_name
        self.active = False
        self.subscription_ids: List[str] = []
        self.metrics: Dict[str, float] = {}
        logger.info(f"xApp initialized: {xapp_name} ({xapp_id})")
    
    def start(self):
        """Start the xApp"""
        self.active = True
        self.on_start()
        logger.info(f"xApp started: {self.xapp_name}")
    
    def stop(self):
        """Stop the xApp"""
        self.active = False
        self.on_stop()
        logger.info(f"xApp stopped: {self.xapp_name}")
    
    def on_start(self):
        """Callback when xApp starts (to be implemented by subclass)"""
        pass
    
    def on_stop(self):
        """Callback when xApp stops (to be implemented by subclass)"""
        pass
    
    def handle_indication(self, indication_data: Dict[str, Any]):
        """
        Handle indication from RAN
        
        Args:
            indication_data: Indication data from E2 interface
        """
        raise NotImplementedError("xApp must implement handle_indication()")
    
    def make_control_decision(self, ran_state: Dict[str, Any]) -> Optional[Dict]:
        """
        Make control decision based on RAN state
        
        Args:
            ran_state: Current RAN state
            
        Returns:
            dict: Control action to be sent to RAN, or None
        """
        raise NotImplementedError("xApp must implement make_control_decision()")
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update xApp performance metrics"""
        self.metrics.update(metrics)


class NearRTRIC:
    """
    Near-Real-Time RAN Intelligent Controller
    
    Manages xApps and E2 connections to RAN nodes.
    Provides real-time RAN optimization with sub-10ms latency.
    """
    
    def __init__(self, config: RICConfig):
        """
        Initialize Near-RT RIC
        
        Args:
            config: RIC configuration
        """
        self.config = config
        self.xapps: Dict[str, xApp] = {}
        self.e2_connections: Dict[str, E2Interface] = {}
        self.control_loop_active = False
        self.control_thread: Optional[threading.Thread] = None
        self.indication_queue = Queue()
        
        # Performance metrics
        self.metrics = {
            "control_loop_latency_ms": [],
            "xapp_execution_time_ms": {},
            "total_control_actions": 0,
            "successful_actions": 0
        }
        
        logger.info(f"Near-RT RIC initialized: {config.ric_id}")
    
    def start(self):
        """Start the Near-RT RIC"""
        logger.info("Starting Near-RT RIC...")
        self.control_loop_active = True
        
        # Start control loop thread
        self.control_thread = threading.Thread(
            target=self._control_loop,
            daemon=True
        )
        self.control_thread.start()
        
        logger.info("Near-RT RIC started successfully")
    
    def stop(self):
        """Stop the Near-RT RIC"""
        logger.info("Stopping Near-RT RIC...")
        self.control_loop_active = False
        
        if self.control_thread:
            self.control_thread.join(timeout=5.0)
        
        # Stop all xApps
        for xapp in self.xapps.values():
            xapp.stop()
        
        # Disconnect all E2 connections
        for e2_conn in self.e2_connections.values():
            e2_conn.disconnect()
        
        logger.info("Near-RT RIC stopped")
    
    def register_xapp(self, xapp: xApp) -> bool:
        """
        Register an xApp with the RIC
        
        Args:
            xapp: xApp instance to register
            
        Returns:
            bool: True if registration successful
        """
        if len(self.xapps) >= self.config.max_xapps:
            logger.error(f"Maximum xApps limit reached: {self.config.max_xapps}")
            return False
        
        if xapp.xapp_id in self.xapps:
            logger.warning(f"xApp already registered: {xapp.xapp_id}")
            return False
        
        self.xapps[xapp.xapp_id] = xapp
        self.metrics["xapp_execution_time_ms"][xapp.xapp_id] = []
        
        # Start the xApp
        xapp.start()
        
        logger.info(f"xApp registered: {xapp.xapp_name}")
        return True
    
    def unregister_xapp(self, xapp_id: str) -> bool:
        """
        Unregister an xApp
        
        Args:
            xapp_id: xApp identifier
            
        Returns:
            bool: True if successful
        """
        if xapp_id not in self.xapps:
            return False
        
        xapp = self.xapps[xapp_id]
        xapp.stop()
        del self.xapps[xapp_id]
        
        logger.info(f"xApp unregistered: {xapp.xapp_name}")
        return True
    
    def connect_e2_node(self, node_config: E2NodeConfig) -> bool:
        """
        Establish E2 connection with RAN node
        
        Args:
            node_config: E2 node configuration
            
        Returns:
            bool: True if connection successful
        """
        node_id = node_config.node_id
        
        if node_id in self.e2_connections:
            logger.warning(f"E2 node already connected: {node_id}")
            return False
        
        e2_interface = E2Interface(node_config)
        
        # Connect to the node
        ric_endpoint = f"{self.config.ric_id}:{self.config.e2_port}"
        if e2_interface.connect(ric_endpoint):
            self.e2_connections[node_id] = e2_interface
            logger.info(f"E2 node connected: {node_id}")
            return True
        
        return False
    
    def disconnect_e2_node(self, node_id: str) -> bool:
        """
        Disconnect E2 node
        
        Args:
            node_id: Node identifier
            
        Returns:
            bool: True if successful
        """
        if node_id not in self.e2_connections:
            return False
        
        self.e2_connections[node_id].disconnect()
        del self.e2_connections[node_id]
        
        logger.info(f"E2 node disconnected: {node_id}")
        return True
    
    def _control_loop(self):
        """
        Main control loop running at configured interval
        
        Executes xApps and processes control decisions in near-real-time
        """
        loop_interval_s = self.config.control_loop_interval_ms / 1000.0
        
        while self.control_loop_active:
            loop_start = time.time()
            
            try:
                # Collect RAN state from all E2 connections
                ran_states = self._collect_ran_states()
                
                # Execute each active xApp
                for xapp_id, xapp in self.xapps.items():
                    if not xapp.active:
                        continue
                    
                    xapp_start = time.time()
                    
                    try:
                        # Make control decision
                        control_action = xapp.make_control_decision(ran_states)
                        
                        # Send control action if decision made
                        if control_action:
                            self._send_control_action(control_action)
                            self.metrics["total_control_actions"] += 1
                            self.metrics["successful_actions"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error executing xApp {xapp.xapp_name}: {e}")
                    
                    # Track xApp execution time
                    xapp_time = (time.time() - xapp_start) * 1000
                    self.metrics["xapp_execution_time_ms"][xapp_id].append(xapp_time)
                
                # Process any pending indications
                self._process_indications()
                
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
            
            # Calculate loop latency
            loop_latency = (time.time() - loop_start) * 1000
            self.metrics["control_loop_latency_ms"].append(loop_latency)
            
            # Maintain control loop timing
            elapsed = time.time() - loop_start
            sleep_time = max(0, loop_interval_s - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(
                    f"Control loop overrun: {elapsed*1000:.2f}ms "
                    f"(target: {self.config.control_loop_interval_ms}ms)"
                )
    
    def _collect_ran_states(self) -> Dict[str, Any]:
        """
        Collect current state from all connected RAN nodes
        
        Returns:
            dict: Aggregated RAN state
        """
        states = {}
        
        for node_id, e2_conn in self.e2_connections.items():
            try:
                states[node_id] = e2_conn.get_statistics()
            except Exception as e:
                logger.error(f"Error collecting state from {node_id}: {e}")
        
        return states
    
    def _send_control_action(self, control_action: Dict[str, Any]):
        """
        Send control action to RAN node
        
        Args:
            control_action: Control action to send
        """
        node_id = control_action.get("node_id")
        
        if node_id not in self.e2_connections:
            logger.error(f"E2 node not connected: {node_id}")
            return
        
        # Create control message
        control_msg = E2Message(
            message_type=E2MessageType.CONTROL_REQUEST,
            transaction_id=f"ctrl_{int(time.time()*1000)}",
            timestamp=time.time(),
            payload=control_action
        )
        
        # Send via E2 interface
        self.e2_connections[node_id].handle_control_request(control_msg)
    
    def _process_indications(self):
        """Process indication messages from RAN nodes"""
        while not self.indication_queue.empty():
            indication = self.indication_queue.get()
            
            # Route indication to interested xApps
            for xapp in self.xapps.values():
                if xapp.active:
                    try:
                        xapp.handle_indication(indication)
                    except Exception as e:
                        logger.error(
                            f"Error in xApp {xapp.xapp_name} "
                            f"handling indication: {e}"
                        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get RIC performance metrics
        
        Returns:
            dict: Performance metrics
        """
        latencies = self.metrics["control_loop_latency_ms"]
        
        metrics = {
            "active_xapps": len([x for x in self.xapps.values() if x.active]),
            "connected_nodes": len(self.e2_connections),
            "total_control_actions": self.metrics["total_control_actions"],
            "success_rate": (
                self.metrics["successful_actions"] / 
                max(1, self.metrics["total_control_actions"])
            )
        }
        
        if latencies:
            metrics.update({
                "avg_control_loop_latency_ms": np.mean(latencies[-100:]),
                "max_control_loop_latency_ms": np.max(latencies[-100:]),
                "p95_control_loop_latency_ms": np.percentile(latencies[-100:], 95)
            })
        
        # xApp-specific metrics
        for xapp_id, times in self.metrics["xapp_execution_time_ms"].items():
            if times:
                xapp_name = self.xapps[xapp_id].xapp_name
                metrics[f"{xapp_name}_avg_execution_ms"] = np.mean(times[-100:])
        
        return metrics
