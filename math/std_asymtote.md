---
title: "STD asymptote analysis: why $b_E$ hits a $0.2$ floor"
author: "train-srnn"
date: "2026-04-30"
---

# STD asymptote analysis: why $b_E$ hits a $0.2$ floor

## Setup

Consider a single excitatory neuron in the `srnn-std-e-only` preset
(\texttt{train\_srnn/models/srnn\_cell.py:135}) with all initial-condition
parameter values held at their defaults. The relevant cell-internal flags
for this preset are
\begin{align*}
n_{a,E} = 0, \quad n_{a,I} = 0, \quad n_{b,E} = 1, \quad n_{b,I} = 0,
\end{align*}
so spike-frequency adaptation (SFA) is disabled and only short-term
synaptic depression (STD) is active, on the excitatory side. We isolate a
single neuron driven by a constant external input $u$, taking the
recurrent contribution $W_{\text{eff}}\,(b \cdot r)$ to be zero (or
absorbed into $u$) for the cleanest closed-form analysis. We address the
recurrent contribution at the end.

The dendritic state of the neuron is $x(t)$, the synaptic-depression
state is $b(t) \in [0,1]$ (we drop the $E$ subscript for the rest of the
analysis), the instantaneous firing rate is $r(t) \in [0,1]$, and the
firing threshold is $a_0$.

The key default parameter values, from
\texttt{srnn\_cell.py} initialization (lines 215, 224, 278--279), are
\begin{align*}
\tau_d            &= 0.10 \text{ s} \quad &\text{(dendritic membrane time constant)} \\
\tau_{\text{rec}} &= 1.00 \text{ s} \quad &\text{(STD recovery)} \\
\tau_{\text{rel}} &= 0.25 \text{ s} \quad &\text{(STD release / depression)} \\
a_0               &= 0.35           \quad &\text{(firing threshold)}.
\end{align*}

The activation function is the piecewise sigmoid $\sigma(\cdot)$ defined
in \texttt{srnn\_cell.py:40--75}, with output bounded in $[0,1]$ and
saturating at $1$ for input $> 0.55$.

## 1. Dendritic dynamics

From \texttt{\_compute\_rhs} (\texttt{srnn\_cell.py:484--485}), the dendritic
ODE in the single-neuron case (no recurrence) is
\begin{align}
\tau_d \, \frac{dx}{dt} \;=\; -x + u + W_{\text{eff}} (b \cdot r).
\end{align}
With identity input drive of constant amplitude $u$ and zero recurrence,
this reduces to
\begin{align}
\tau_d \, \frac{dx}{dt} \;=\; -x + u,
\end{align}
which is a first-order linear ODE with solution
\begin{align}
x(t) \;=\; u + \big(x(0) - u\big) e^{-t/\tau_d}.
\end{align}
For $x(0) \approx 0$ (the cell initializes the dendrite at $0.1 \cdot
\mathcal{N}(0,1)$, so $|x(0)|$ is small), the dendrite settles to
$x_{\infty} = u$ on a timescale of order $\tau_d = 0.1$ s. After roughly
$5\tau_d = 0.5$ s, $x \approx u$ to within $1\%$.

## 2. Firing rate

The firing rate is computed in \texttt{srnn\_cell.py:470} as
\begin{align}
r(t) \;=\; \sigma\!\big(x(t) - a_0\big).
\end{align}
The piecewise sigmoid $\sigma$ saturates at $r = 1$ once its argument
exceeds $0.55$. So firing saturates at $r=1$ once
\begin{align}
x(t) - a_0 \;>\; 0.55, \qquad \text{i.e.} \qquad x(t) \;>\; 0.90.
\end{align}
For a step input $u$ with $u > 0.9$, the dendrite settles to $x \approx u
> 0.9$ after the transient, and $r$ then saturates at $r = 1$
indefinitely.

The crucial observation is that $r$ is bounded above by $1$. No matter
how strong the input, the firing rate cannot exceed unity, because the
activation function caps it. This bound is what produces the floor on
$b$ derived below.

## 3. STD dynamics

From \texttt{\_compute\_rhs} (\texttt{srnn\_cell.py:505}), the STD ODE for
the excitatory neuron is
\begin{align}
\frac{db}{dt} \;=\; \frac{1 - b}{\tau_{\text{rec}}} \;-\; \frac{r \cdot
b}{\tau_{\text{rel}}}.
\label{eq:b_ode}
\end{align}
Two opposing terms:
\begin{itemize}
\item the recovery term $(1 - b)/\tau_{\text{rec}}$ pulls $b$ back toward
$1$ (no depression);
\item the release term $r \cdot b / \tau_{\text{rel}}$ depresses $b$ at a
rate proportional to both the current firing rate $r$ and the current
depression level $b$.
\end{itemize}
At $b = 1$ (rested) the recovery is zero and depression dominates as
soon as $r > 0$. At $b = 0$ the depression is zero and recovery
dominates. The dynamics are therefore a competition between a linear
restoring force and a multiplicative depletion.

## 4. Steady-state derivation

Assume the dendrite has settled and $r$ is constant ($r = r_\infty$).
Setting $db/dt = 0$ in equation~\eqref{eq:b_ode} gives
\begin{align}
0 \;&=\; \frac{1 - b_\infty}{\tau_{\text{rec}}} \;-\; \frac{r_\infty \cdot
b_\infty}{\tau_{\text{rel}}}, \\
\frac{1 - b_\infty}{\tau_{\text{rec}}} \;&=\; \frac{r_\infty \cdot
b_\infty}{\tau_{\text{rel}}}, \\
\tau_{\text{rel}} \,(1 - b_\infty) \;&=\; \tau_{\text{rec}} \, r_\infty \,
b_\infty, \\
\tau_{\text{rel}} \;&=\; b_\infty \,\big(\tau_{\text{rec}} \, r_\infty +
\tau_{\text{rel}}\big), \\
\boxed{\;b_\infty(r_\infty) \;=\; \frac{\tau_{\text{rel}}}{\tau_{\text{rec}}
\, r_\infty + \tau_{\text{rel}}}.\;}
\label{eq:b_inf}
\end{align}
This is the Tsodyks--Markram steady-state expression; equation~\eqref{eq:b_inf}
is monotonically decreasing in $r_\infty$.

## 5. Numerical asymptote at the IC parameter values

Plugging the default values $\tau_{\text{rec}} = 1.0$ s and
$\tau_{\text{rel}} = 0.25$ s into equation~\eqref{eq:b_inf} as a function
of $r_\infty$:
\begin{align*}
b_\infty(r_\infty) \;=\; \frac{0.25}{1.0 \cdot r_\infty + 0.25}
\;=\; \frac{1}{4 r_\infty + 1}.
\end{align*}
Selected values:
\begin{center}
\begin{tabular}{r|l}
$r_\infty$ & $b_\infty$ \\
\hline
$0.0$  & $1.000$ \\
$0.1$  & $0.714$ \\
$0.25$ & $0.500$ \\
$0.5$  & $0.333$ \\
$0.75$ & $0.250$ \\
$1.0$  & $\mathbf{0.200}$
\end{tabular}
\end{center}

Because $r_\infty \in [0, 1]$ (bounded by the saturating piecewise sigmoid
in section 2), the minimum of $b_\infty(r_\infty)$ over the allowed
range is achieved at $r_\infty = 1$ and equals
\begin{align}
\boxed{\;b_\infty^{\min} \;=\; \frac{\tau_{\text{rel}}}{\tau_{\text{rec}}
+ \tau_{\text{rel}}} \;=\; \frac{0.25}{1.25} \;=\; 0.20.\;}
\label{eq:b_floor}
\end{align}
This is the asymptotic floor: a single neuron driven hard enough to
saturate its firing rate cannot depress its synaptic-depression state
$b$ below $0.20$ at the IC parameter values, no matter how long the
input persists.

The threshold $b_\infty = 0.5$ is hit at the modest firing rate
$r_\infty = \tau_{\text{rel}}/\tau_{\text{rec}} = 0.25$. Any neuron
firing at $r > 0.25$ will sustain $b < 0.5$ at steady state. So crossing
the $0.5$ line in numerical experiments is easy; the $0.2$ floor is the
hard limit set by full saturation.

## 6. Time constant of approach

For constant $r_\infty$, equation~\eqref{eq:b_ode} is linear in $b$:
\begin{align}
\frac{db}{dt} \;=\; \frac{1}{\tau_{\text{rec}}} \;-\; b \,\Big(\frac{1}{\tau_{\text{rec}}}
+ \frac{r_\infty}{\tau_{\text{rel}}}\Big).
\end{align}
This has the form $\dot{b} = -\big(b - b_\infty\big)/\tau_{\text{eff}}$, with
\begin{align}
\frac{1}{\tau_{\text{eff}}} \;=\; \frac{1}{\tau_{\text{rec}}}
+ \frac{r_\infty}{\tau_{\text{rel}}},
\qquad
\tau_{\text{eff}}(r_\infty) \;=\; \frac{\tau_{\text{rec}}\,\tau_{\text{rel}}}
{\tau_{\text{rel}} + r_\infty\,\tau_{\text{rec}}}.
\label{eq:tau_eff}
\end{align}
At the IC parameter values:
\begin{center}
\begin{tabular}{r|l}
$r_\infty$ & $\tau_{\text{eff}}$ (s) \\
\hline
$0.0$  & $1.000$ \\
$0.25$ & $0.500$ \\
$0.5$  & $0.333$ \\
$1.0$  & $\mathbf{0.200}$
\end{tabular}
\end{center}
At full saturation, $\tau_{\text{eff}} = 0.2$ s, so depression is
essentially complete after $5\tau_{\text{eff}} = 1.0$ s of sustained
firing. This matches the empirical observation in
\texttt{scripts/diag\_std\_b\_e\_step.py} that $b$ reaches the $0.2$
asymptote within the first second of a strong step.

The closed-form trajectory for a step input applied at $t=0$, with $b(0)
= 1$, is
\begin{align}
b(t) \;=\; b_\infty + (1 - b_\infty)\, e^{-t/\tau_{\text{eff}}}.
\end{align}

## 7. Why $r_\infty < 1$ is the typical case

The floor $b_\infty^{\min} = 0.2$ is reached only when the neuron is
driven hard enough to saturate its firing rate. In a realistic network
this is often not the case:
\begin{itemize}
\item Recurrent inhibition (Dale's law gives an explicit I population)
suppresses E firing, so a typical E neuron in the trained network
operates at $r$ well below $1$.
\item For zero-mean inputs (as with z-scored SEEG channels), driving
$x$ above the saturation threshold $x > 0.9$ requires either large
positive excursions in the input or strong recurrent excitation.
\item Once trained, $a_0$ may have shifted away from its initial value
$0.35$, raising or lowering the effective threshold for saturation.
\end{itemize}
At the modest typical firing rates achieved during SEEG training
(empirically order $r \approx 0.1$--$0.3$), the steady-state $b$ from
equation~\eqref{eq:b_inf} is in the range $0.45$--$0.71$, all above
$0.5$. This is consistent with the observation that trained models'
$b_E$ rarely depresses below $0.5$ in practice, despite the cell
correctly producing a $0.2$ asymptote when given a strong artificial
step (see the diagnostic in
\texttt{scripts/diag\_std\_b\_e\_step.py}).

## 8. Effect of recurrent contributions

Restoring the recurrent term in the dendrite ODE gives
\begin{align}
\tau_d \,\frac{dx}{dt} \;=\; -x + u + W_{\text{eff}}\,(b \cdot r).
\end{align}
For a single neuron embedded in the network, $W_{\text{eff}}\,(b \cdot
r)$ contributes an additional drive that can either push $x$ above the
saturation threshold (recurrent excitation) or pull it down (recurrent
inhibition via Dale's law). The qualitative analysis is unchanged:
\begin{itemize}
\item $r$ remains bounded in $[0,1]$ since it is the output of $\sigma$;
\item the steady-state $b_\infty(r)$ from equation~\eqref{eq:b_inf}
still applies, with $r$ taking the value induced by the combined
external + recurrent drive at steady state;
\item the floor $b_\infty^{\min} = 0.2$ from
equation~\eqref{eq:b_floor} is unchanged.
\end{itemize}
Recurrence shifts where on the curve $b_\infty(r_\infty)$ a given
neuron operates, but does not lower the floor.

## 9. Lower-bound interpretation

The $0.2$ floor has a clean physical interpretation:
$\tau_{\text{rel}}/(\tau_{\text{rec}} + \tau_{\text{rel}})$ is the
fraction of synaptic resources that, at maximum sustained activity, are
in a usable (not depressed) state. With the IC values, $1/5$ of
resources remain available even when the neuron is firing at maximum
rate, because the recovery process operates at a finite rate
($1/\tau_{\text{rec}} = 1$ Hz at default) regardless of how strong the
release is.

To enable $b$ to fall below $0.2$, one of the following would have to
change:
\begin{itemize}
\item Lower $\tau_{\text{rel}}$ relative to $\tau_{\text{rec}}$ (faster
release): e.g. $\tau_{\text{rel}} = 0.05$ would give floor $0.05/1.05
\approx 0.048$.
\item Increase $\tau_{\text{rec}}$ relative to $\tau_{\text{rel}}$
(slower recovery): e.g. $\tau_{\text{rec}} = 4$ would give floor
$0.25/4.25 \approx 0.059$.
\item Allow $r$ to exceed $1$ (e.g. by replacing the bounded sigmoid
with an unbounded activation), in which case
$b_\infty = \tau_{\text{rel}}/(\tau_{\text{rec}} \cdot r +
\tau_{\text{rel}})$ would continue decreasing past $0.2$.
\end{itemize}
None of these are happening at IC parameter values; hence the floor.

## 10. Summary

\begin{itemize}
\item The STD ODE in the cell is the standard Tsodyks--Markram form
$\dot{b} = (1-b)/\tau_{\text{rec}} - r\,b/\tau_{\text{rel}}$.
\item Steady-state: $b_\infty(r) =
\tau_{\text{rel}}/(\tau_{\text{rec}}\, r + \tau_{\text{rel}})$.
\item At IC values $\tau_{\text{rec}} = 1.0,\ \tau_{\text{rel}} = 0.25$,
$b_\infty(1) = 0.20$ is the minimum.
\item Approach time constant at $r=1$: $\tau_{\text{eff}} = 0.20$ s.
Full depression in $\sim 1$ s.
\item The bounded firing rate $r \in [0,1]$ from the piecewise sigmoid
caps depression at this floor.
\item Trained models that operate at $r \ll 1$ on average never reach
this floor, even though the underlying ODE permits it (this is what
the diagnostic script confirms numerically).
\end{itemize}
