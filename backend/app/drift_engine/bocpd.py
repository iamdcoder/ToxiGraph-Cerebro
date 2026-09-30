import math


class BayesianChangePointDetector:
    """
    Bayesian Online Change-Point Detection.

    The detector maintains a posterior distribution over run
    lengths. A run length describes how many observations have
    occurred since the most recent regime change.

    Two probabilities are exposed:

    1. run_length_zero_probability
       The exact posterior probability at run length 0.

       With a constant hazard this can remain close to the
       hazard prior, so it is not used as CEREBRO's main
       change signal.

    2. recent_change_probability
       Posterior mass over short run lengths.

       This is the data-driven signal used by CEREBRO to detect
       recent emotional regime changes.
    """

    def __init__(
        self,
        hazard_rate: float = 1 / 50,
        mu_prior: float = 0.0,
        kappa_prior: float = 1.0,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        max_run_length: int = 100,
        recent_window: int = 5,
    ) -> None:

        if not 0 < hazard_rate < 1:
            raise ValueError(
                "hazard_rate must be between 0 and 1."
            )

        if recent_window < 0:
            raise ValueError(
                "recent_window must be >= 0."
            )

        self.hazard = hazard_rate

        self.prior = (
            mu_prior,
            kappa_prior,
            alpha_prior,
            beta_prior,
        )

        self.max_run_length = max_run_length
        self.recent_window = recent_window

        # P(r_t)
        self.run_length_probs = [1.0]

        # Sufficient statistics for each run length.
        self.parameters = [self.prior]

        # Exact P(r_t = 0).
        self.change_point_probs: list[float] = []

        # Data-driven recent change probability:
        # P(r_t <= recent_window).
        self.recent_change_probs: list[float] = []

        self.last_run_length_zero_probability = 1.0
        self.last_recent_change_probability = 1.0

    @staticmethod
    def _student_t_pdf(
        x: float,
        mu: float,
        kappa: float,
        alpha: float,
        beta: float,
    ) -> float:

        degrees_of_freedom = 2.0 * alpha

        scale = math.sqrt(
            max(
                beta
                * (kappa + 1.0)
                / (alpha * kappa),
                1e-12,
            )
        )

        z = (x - mu) / scale

        log_pdf = (
            math.lgamma(
                (degrees_of_freedom + 1.0) / 2.0
            )
            - math.lgamma(
                degrees_of_freedom / 2.0
            )
            - 0.5
            * math.log(
                degrees_of_freedom * math.pi
            )
            - math.log(scale)
            - (
                (degrees_of_freedom + 1.0)
                / 2.0
            )
            * math.log1p(
                (z * z)
                / degrees_of_freedom
            )
        )

        if log_pdf < -690:
            return 1e-300

        return math.exp(log_pdf)

    @staticmethod
    def _update_parameters(
        parameters: tuple[float, float, float, float],
        x: float,
    ) -> tuple[float, float, float, float]:

        mu, kappa, alpha, beta = parameters

        new_kappa = kappa + 1.0

        new_mu = (
            kappa * mu + x
        ) / new_kappa

        new_alpha = alpha + 0.5

        new_beta = beta + (
            0.5
            * kappa
            / new_kappa
            * (x - mu) ** 2
        )

        return (
            new_mu,
            new_kappa,
            new_alpha,
            new_beta,
        )

    def update(
        self,
        observation: float,
    ) -> float:
        """
        Process one new observation.

        Returns:
            Data-driven probability that the current
            observation is close to a recent change,
            represented by posterior mass over short
            run lengths.
        """

        old_probs = self.run_length_probs
        old_parameters = self.parameters

        predictive = [
            max(
                self._student_t_pdf(
                    observation,
                    *parameters,
                ),
                1e-300,
            )
            for parameters in old_parameters
        ]

        # Growth probabilities:
        # r_t = r_(t-1) + 1
        growth_probs = [
            old_probs[index]
            * predictive[index]
            * (1.0 - self.hazard)
            for index in range(
                len(old_probs)
            )
        ]

        # Change-point probability:
        # r_t = 0
        change_probability = sum(
            old_probs[index]
            * predictive[index]
            * self.hazard
            for index in range(
                len(old_probs)
            )
        )

        new_probs = [
            change_probability,
            *growth_probs,
        ]

        # Update posterior parameters.
        new_parameters = [
            self._update_parameters(
                self.prior,
                observation,
            )
        ]

        for parameters in old_parameters:
            new_parameters.append(
                self._update_parameters(
                    parameters,
                    observation,
                )
            )

        # Truncate very old run lengths.
        limit = self.max_run_length + 1

        if len(new_probs) > limit:
            new_probs = new_probs[:limit]
            new_parameters = new_parameters[:limit]

        total_probability = sum(new_probs)

        if total_probability <= 0:
            new_probs = [
                1.0
            ]

            new_parameters = [
                self._update_parameters(
                    self.prior,
                    observation,
                )
            ]

        else:
            new_probs = [
                probability / total_probability
                for probability in new_probs
            ]

        self.run_length_probs = new_probs
        self.parameters = new_parameters

        # Exact run-length-zero posterior.
        run_length_zero_probability = (
            self.run_length_probs[0]
        )

        # Data-driven short-run probability.
        recent_limit = min(
            self.recent_window + 1,
            len(self.run_length_probs),
        )

        recent_change_probability = sum(
            self.run_length_probs[
                :recent_limit
            ]
        )

        # Before enough observations exist to have a meaningful
        # short-run distribution, do not report a false "recent
        # change" probability of 1.0 simply because every possible
        # run length currently falls inside the short window.
        observations_seen = len(
            self.change_point_probs
        )

        if observations_seen <= self.recent_window:
            recent_change_probability = 0.0

        self.last_run_length_zero_probability = (
            run_length_zero_probability
        )

        self.last_recent_change_probability = (
            recent_change_probability
        )

        self.change_point_probs.append(
            run_length_zero_probability
        )

        self.recent_change_probs.append(
            recent_change_probability
        )

        return recent_change_probability

    def most_likely_run_length(self) -> int:
        if not self.run_length_probs:
            return 0

        return max(
            range(len(self.run_length_probs)),
            key=lambda index: self.run_length_probs[index],
        )

    def reset(self) -> None:
        self.run_length_probs = [1.0]

        self.parameters = [
            self.prior
        ]

        self.change_point_probs = []

        self.recent_change_probs = []

        self.last_run_length_zero_probability = 1.0

        self.last_recent_change_probability = 1.0