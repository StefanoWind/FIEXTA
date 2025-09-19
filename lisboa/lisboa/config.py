from dataclasses import dataclass

@dataclass
class LisboaConfig:
    """Configuration parameters for LiSBOA."""
    sigma: float=0.25
    mins: list = (-1,-1)
    maxs: list = (1,1)
    Dn0: list = (1,1)
    r_max: float=3
    dist_edge: float=1
    tol_dist: float=0.1
    grid_factor: float=0.25
    max_Dd: float= 1,
    max_iter: float= 5
    
    def _validate_dims(self,  mins: list, maxs: list, Dn0) -> None:
        """Validate dimensions."""
        if not len(mins)==len(maxs)==len(Dn0):
            raise ValueError("mins, maxs and Dn0 must have the same number of elements")
        else:
            if len(mins)==1:
                raise ValueError("mins, maxs and Dn0 musth have at least 2 elements")
    
    def _validate_sigma(self, sigma: float) -> None:
        """Validate sigma model."""
        if sigma<=0:
            raise ValueError(f"sigma must be positive (given value is {sigma})")
    
    def _validate_mins_maxs(self, mins: list, maxs: list) -> None:
        """Validate mins and maxs."""
        for m,M in zip(mins,maxs):
            if m>=M:
                raise ValueError(f"mins must be stricitly lower than maxs (one value in mins is {m} and corresponding value in maxs is {M})")
    
    def _validate_r_max(self, r_max: float) -> None:
        """Validate r_max model."""
        if r_max<=0:
            raise ValueError(f"R_max must be positive (given value is {r_max})")
            
    def _validate_dist_edge(self, dist_edge: float) -> None:
        """Validate dist_edge model."""
        if dist_edge<=0:
            raise ValueError(f"dist_edge must be positive (given value is {dist_edge})")
            
    def _validate_Dn0(self, Dn0: list) -> None:
        """Validate Dn0"""
        for d in Dn0:
            if d<=0:
                raise ValueError(f"Every element of Dn0 must be positive (one value is {d})")
                
    def _validate_tol_dist(self, tol_dist: float) -> None:
        """Validate tol_dist model."""
        if tol_dist<=0:
            raise ValueError(f"tol_dist must be positive (given value is {tol_dist})")
            
    def _validate_grid_factor(self, grid_factor: float) -> None:
        """Validate tol_dist model."""
        if grid_factor<=0:
            raise ValueError(f"grid_factor must be positive (given value is {grid_factor})")
            
    def _validate_max_Dd(self, max_Dd: float) -> None:
        """Validate tol_dist model."""
        if max_Dd<=0:
            raise ValueError(f"max_Dd must be positive (given value is {max_Dd})")
            
    def _validate_max_iter(self, max_iter: int) -> None:
        """Validate tol_dist model."""
        if max_iter<=0 or max_iter!=int(max_iter):
            raise ValueError(f"max_iter must be a positive interger (given value is {max_iter})")
        

    def validate(self) -> None:
        """Validate all configuration parameters."""
        # Validate dates
        self._validate_dims(self.mins, self.maxs,self.Dn0)
        self._validate_mins_maxs(self.mins, self.maxs)
        self._validate_sigma(self.sigma)
        self._validate_r_max(self.r_max)
        self._validate_dist_edge(self.dist_edge)
        self._validate_Dn0(self.Dn0)
        self._validate_tol_dist(self.tol_dist)
        self._validate_grid_factor(self.grid_factor)
        self._validate_max_Dd(self.max_Dd)
        self._validate_max_iter(self.max_iter)
