# From equation (2) to equation (3)

Equation (3) is equation (2) after the joint intensity is split into a rate of events and a magnitude density. The magnitude integral then equals 1 and drops out of the compensator.

## Equation (2)

Equation (2) is the log-likelihood of the observed times and magnitudes:

$$\log L=\sum_i\left[\log\lambda(t_i,m_i\mid H_{t_i})-\int_{t_{i-1}}^{t_i}\int_{\mathcal{M}}\lambda(t,m\mid H_t)\,dm\,dt\right].$$

The first term scores the event that was observed. The double integral is the compensator: the expected number of events of every magnitude between $t_{i-1}$ and $t_i$.

## The factorization

The joint intensity is written as

$$\lambda^*(t,m)=\lambda^*(t)\,f^*(m\mid t).$$

$\lambda^*(t)$ is the rate of an earthquake at time $t$, of any magnitude. $f^*(m\mid t)$ is the probability density of its magnitude, given that an earthquake occurs at $t$. Both still depend on the history $H_t$.

## Substitute into the log term

The observed event splits into two scores:

$$\log\lambda(t_i,m_i\mid H_{t_i})=\log\lambda^*(t_i)+\log f^*(m_i\mid t_i).$$

## Substitute into the compensator

The double integral becomes

$$\int_{t_{i-1}}^{t_i}\lambda^*(t)\left(\int_{\mathcal{M}}f^*(m\mid t)\,dm\right)dt.$$

At each fixed time, $f^*(\cdot\mid t)$ is a probability density, so

$$\int_{\mathcal{M}}f^*(m\mid t)\,dm=1.$$

The inner integral is therefore 1 even when the magnitude distribution changes with time and with past magnitudes. The compensator reduces to

$$\int_{t_{i-1}}^{t_i}\lambda^*(t)\,dt.$$
why 
## Equation (3)

$$\log L=\sum_i\left[\log\lambda^*(t_i)+\log f^*(m_i\mid t_i)-\int_{t_{i-1}}^{t_i}\lambda^*(t)\,dt\right].$$

The compensator counts how many earthquakes are expected in the gap, whatever their magnitude would have been. A magnitude is attached only to an earthquake that actually occurs, so it is scored only in $\log f^*(m_i\mid t_i)$. Past magnitudes still affect both $\lambda^*(t)$ and $f^*$ through the history.
