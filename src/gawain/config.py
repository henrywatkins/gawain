"""Configuration validation using Pydantic

This module provides Pydantic models for validating simulation configuration
parameters with comprehensive type checking and validation rules.
"""

from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class IntegratorType(str, Enum):
    """Available integrator types"""

    EULER = "euler"


class FluxerType(str, Enum):
    """Available fluxer types"""

    BASE = "base"
    LAX_WENDROFF = "lax-wendroff"
    LAX_FRIEDRICHS = "lax-friedrichs"
    HLL = "hll"


class BoundaryType(str, Enum):
    """Available boundary condition types"""

    PERIODIC = "periodic"
    FIXED = "fixed"
    REFLECTIVE = "reflective"


class SimulationConfig(BaseModel):
    """Pydantic model for simulation configuration validation

    This model validates all simulation parameters with appropriate
    type checking, bounds validation, and consistency checks.
    """

    # Basic simulation parameters
    run_name: str = Field(..., min_length=1, description="Name of the simulation run")

    cfl: float = Field(
        ..., gt=0.0, le=1.0, description="CFL condition parameter (0 < cfl <= 1)"
    )

    t_max: float = Field(..., gt=0.0, description="Maximum simulation time")

    n_dumps: int = Field(..., ge=1, description="Number of output dumps")

    output_dir: Union[str, Path] = Field(..., description="Directory for output files")

    # Physics parameters
    adi_idx: float = Field(
        ..., gt=1.0, description="Adiabatic index (gamma), must be > 1"
    )

    with_mhd: bool = Field(..., description="Whether to include MHD terms")

    # Numerical methods
    integrator: IntegratorType = Field(..., description="Time integration method")

    fluxer: FluxerType = Field(..., description="Flux calculation method")

    # Mesh parameters
    mesh_shape: Tuple[int, int, int] = Field(
        ..., description="Number of cells in each direction (nx, ny, nz)"
    )

    mesh_size: Tuple[float, float, float] = Field(
        ..., description="Physical size of domain in each direction (lx, ly, lz)"
    )

    # Boundary conditions
    boundary_type: List[BoundaryType] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Boundary conditions for each axis [x, y, z]",
    )

    # Initial conditions and optional fields
    initial_condition: Optional[np.ndarray] = Field(
        None, description="Initial condition array"
    )

    mesh_grid: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = Field(
        None, description="Mesh grid arrays (X, Y, Z)"
    )

    source: Optional[np.ndarray] = Field(None, description="Source term array")

    gravity: Optional[np.ndarray] = Field(None, description="Gravity field array")

    class Config:
        # Allow numpy arrays and other arbitrary types
        arbitrary_types_allowed = True
        # Don't validate assignment after model creation
        validate_assignment = True

    @field_validator("mesh_shape")
    @classmethod
    def validate_mesh_shape(cls, v: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Validate mesh shape parameters"""
        nx, ny, nz = v
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("All mesh dimensions must be >= 1")
        return v

    @field_validator("mesh_size")
    @classmethod
    def validate_mesh_size(
        cls, v: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Validate mesh size parameters"""
        lx, ly, lz = v
        if lx <= 0 or ly <= 0 or lz <= 0:
            raise ValueError("All mesh sizes must be > 0")
        return v

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Union[str, Path]) -> str:
        """Convert Path to string and validate output directory"""
        return str(v)

    @field_validator("initial_condition")
    @classmethod
    def validate_initial_condition(
        cls, v: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Validate initial condition array"""
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise ValueError("initial_condition must be a numpy array")
            if v.ndim != 4:
                raise ValueError(
                    "initial_condition must be a 4D array [variables, nx, ny, nz]"
                )
        return v

    @field_validator("mesh_grid")
    @classmethod
    def validate_mesh_grid(
        cls, v: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Validate mesh grid arrays"""
        if v is not None:
            X, Y, Z = v
            if not all(isinstance(arr, np.ndarray) for arr in [X, Y, Z]):
                raise ValueError("All mesh grid components must be numpy arrays")
        return v

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Validate source term array"""
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise ValueError("source must be a numpy array")
        return v

    @field_validator("gravity")
    @classmethod
    def validate_gravity(cls, v: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Validate gravity field array"""
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise ValueError("gravity must be a numpy array")
            if v.ndim != 4:
                raise ValueError("gravity must be a 4D array [3, nx, ny, nz]")
            if v.shape[0] != 3:
                raise ValueError("gravity field must have 3 components")
        return v

    @model_validator(mode="after")
    def validate_array_consistency(self) -> "SimulationConfig":
        """Validate consistency between arrays and mesh parameters"""
        if self.initial_condition is not None:
            expected_shape = self.mesh_shape
            actual_shape = self.initial_condition.shape[1:]  # Skip variable dimension
            if actual_shape != expected_shape:
                raise ValueError(
                    f"initial_condition mesh shape {actual_shape} does not match "
                    f"specified mesh_shape {expected_shape}"
                )

            # Validate number of variables
            n_vars = self.initial_condition.shape[0]
            expected_vars = 8 if self.with_mhd else 5
            if n_vars != expected_vars:
                raise ValueError(
                    f"initial_condition has {n_vars} variables, expected {expected_vars} "
                    f"for {'MHD' if self.with_mhd else 'HD'} simulation"
                )

        if self.mesh_grid is not None:
            X, Y, Z = self.mesh_grid
            expected_shape = self.mesh_shape
            for i, (arr, name) in enumerate(zip([X, Y, Z], ["X", "Y", "Z"])):
                if arr.shape != expected_shape:
                    raise ValueError(
                        f"mesh_grid {name} shape {arr.shape} does not match "
                        f"mesh_shape {expected_shape}"
                    )

        if self.source is not None and self.initial_condition is not None:
            if self.source.shape != self.initial_condition.shape:
                raise ValueError(
                    "source array shape must match initial_condition shape"
                )

        if self.gravity is not None:
            expected_shape = (3,) + self.mesh_shape
            if self.gravity.shape != expected_shape:
                raise ValueError(
                    f"gravity field shape {self.gravity.shape} does not match "
                    f"expected shape {expected_shape}"
                )

        return self

    @model_validator(mode="after")
    def validate_physical_constraints(self) -> "SimulationConfig":
        """Validate physical constraints on initial conditions"""
        if self.initial_condition is not None:
            # Check for negative density
            density = self.initial_condition[0]
            if np.any(density <= 0):
                raise ValueError("Density must be positive everywhere")

            # Check for negative pressure/energy
            energy = self.initial_condition[4]
            momentum_x = self.initial_condition[1]
            momentum_y = self.initial_condition[2]
            momentum_z = self.initial_condition[3]

            # Calculate kinetic energy
            kinetic_energy = (
                0.5 * (momentum_x**2 + momentum_y**2 + momentum_z**2) / density
            )

            # Calculate internal energy (pressure / (gamma - 1))
            internal_energy = energy - kinetic_energy

            if np.any(internal_energy <= 0):
                raise ValueError(
                    "Internal energy (pressure) must be positive everywhere"
                )

        return self


def validate_config(config_dict: dict) -> SimulationConfig:
    """Validate a configuration dictionary using Pydantic

    Parameters
    ----------
    config_dict : dict
        Dictionary containing simulation configuration

    Returns
    -------
    SimulationConfig
        Validated configuration object

    Raises
    ------
    ValidationError
        If configuration is invalid
    """
    return SimulationConfig(**config_dict)
