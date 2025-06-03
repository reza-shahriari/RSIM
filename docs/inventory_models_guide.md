# Inventory Management Models in SimLib

This guide provides comprehensive documentation for the inventory management models available in SimLib. These models help analyze and optimize inventory policies for various business scenarios.

## Overview

SimLib includes four main inventory management models:

1. **(s,S) Inventory Policy** - Continuous review with order-up-to level
2. **(s,Q) Inventory Policy** - Continuous review with fixed order quantity
3. **Newsvendor Model** - Single-period stochastic optimization
4. **EOQ Model** - Economic Order Quantity for deterministic demand

## Model Descriptions

### 1. (s,S) Inventory Policy

The (s,S) policy is a continuous review system where:
- **s** = reorder point (when to order)
- **S** = order-up-to level (target inventory level)
- When inventory drops to or below **s**, order enough to bring inventory up to **S**

**Best for:**
- High-value items with variable demand
- When order quantities can be flexible
- Situations requiring high service levels

**Key Features:**
- Variable order quantities
- Continuous inventory monitoring
- Handles demand and lead time uncertainty
- Optimizes service level vs. cost trade-off

### 2. (s,Q) Inventory Policy

The (s,Q) policy is a continuous review system where:
- **s** = reorder point (when to order)
- **Q** = fixed order quantity (how much to order)
- When inventory drops to or below **s**, order exactly **Q** units

**Best for:**
- Items with supplier quantity constraints
- When order processing is standardized
- Situations with economies of scale in ordering

**Key Features:**
- Fixed order quantities
- Predictable order patterns
- Easier supplier relationship management
- Good for bulk purchasing agreements

### 3. Newsvendor Model

Single-period inventory optimization for perishable or seasonal items:
- Balances overstocking costs vs. understocking costs
- Determines optimal order quantity before demand is known
- Considers salvage value and shortage penalties

**Best for:**
- Seasonal products (fashion, holiday items)
- Perishable goods (food, flowers)
- One-time purchasing decisions
- Products with short selling seasons

**Key Features:**
- Critical fractile analysis
- Profit maximization
- Risk analysis with demand uncertainty
- Sensitivity analysis for key parameters

### 4. EOQ Model

Economic Order Quantity for deterministic demand:
- Minimizes total cost (ordering + holding)
- Assumes constant demand rate
- Provides baseline for more complex models

**Best for:**
- Stable, predictable demand
- Long-term planning
- Cost structure analysis
- Benchmark comparisons

**Key Features:**
- Optimal order quantity calculation
- Cost minimization
- Cycle time analysis
- Sensitivity analysis

## Quick Start Guide

### Installation and Import

```python
from simlib.inventory import SSInventoryPolicy, SQInventoryPolicy, NewsvendorModel, EOQModel
```

### Basic Usage Examples

#### (s,S) Policy Example

```python
# Configure (s,S) policy for electronics retailer
ss_policy = SSInventoryPolicy(
    reorder_point=50,           # Reorder when inventory ≤ 50
    order_up_to_level=200,      # Order up to 200 units
    demand_rate=15,             # 15 units/day average demand
    demand_std=5,               # Standard deviation of 5
    lead_time_mean=7,           # 7 days average lead time
    lead_time_std=2,            # 2 days std dev
    holding_cost=2.0,           # $2/unit/day holding cost
    shortage_cost=25.0,         # $25/unit/day shortage cost
    ordering_cost=100.0,        # $100 per order
    simulation_days=365         # Simulate 1 year
)

# Run simulation
result = ss_policy.run()

# Display results
print(f"Total annual cost: ${result.results['total_cost']:.2f}")
print(f"Service level: {result.results['service_level']:.2%}")
print(f"Average inventory: {result.results['average_inventory']:.1f} units")

# Visualize results
ss_policy.visualize()
```

#### Newsvendor Model Example

```python
# Configure for fashion retailer
newsvendor = NewsvendorModel(
    unit_cost=25.0,             # $25 cost per item
    selling_price=60.0,         # $60 selling price
    salvage_value=10.0,         # $10 salvage value
    shortage_penalty=15.0,      # $15 goodwill loss per unmet demand
    demand_mean=200,            # Expected demand of 200
    demand_std=40,              # Standard deviation of 40
    n_simulations=10000         # 10,000 scenarios
)

# Run optimization
result = newsvendor.run()

# Display results
print(f"Optimal order quantity: {result.results['optimal_quantity']:.0f}")
print(f"Expected profit: ${result.results['expected_profit']:.2f}")
print(f"Critical fractile: {result.results['critical_fractile']:.3f}")

# Visualize
newsvendor.visualize()
```

#### EOQ Model Example

```python
# Configure for warehouse operation
eoq = EOQModel(
    demand_rate=5000,           # 5,000 units/year
    ordering_cost=150,          # $150 per order
    holding_cost_rate=0.25,     # 25% of unit cost/year
    unit_cost=20,               # $20 per unit
    lead_time=14,               # 14 days lead time
    safety_stock=100            # 100 units safety stock
)

# Calculate optimal policy
result = eoq.run()

# Display results
print(f"Optimal order quantity: {result.results['optimal_quantity']:.0f}")
print(f"Minimum annual cost: ${result.results['optimal_cost']:,.2f}")
print(f"Order frequency: {result.results['order_frequency']:.2f} orders/year")

# Visualize
eoq.visualize()
```

## Parameter Guidelines

### Demand Parameters
- **demand_rate**: Average demand per time period
- **demand_std**: Standard deviation of demand (for stochastic models)
- **demand_distribution**: 'normal', 'poisson', 'uniform' (Newsvendor)

### Cost Parameters
- **holding_cost**: Cost to hold one unit for one time period
- **shortage_cost**: Cost of being short one unit for one time period
- **ordering_cost**: Fixed cost per order placed
- **unit_cost**: Purchase cost per unit

### Lead Time Parameters
- **lead_time_mean**: Average lead time
- **lead_time_std**: Standard deviation of lead time
- **lead_time**: Deterministic lead time (EOQ model)

### Policy Parameters
- **reorder_point (s)**: Inventory level that triggers an order
- **order_up_to_level (S)**: Target inventory level for (s,S) policy
- **order_quantity (Q)**: Fixed order quantity for (s,Q) policy
- **safety_stock**: Buffer stock to handle uncertainty

## Model Selection Guide

### Decision Framework

| Scenario | Recommended Model | Rationale |
|----------|------------------|-----------|
| Predictable demand, stable supply | EOQ | Simple, cost-effective |
| Variable demand, flexible ordering | (s,S) Policy | Adapts to demand variability |
| Fixed order constraints | (s,Q) Policy | Respects quantity constraints |
| Seasonal/perishable items | Newsvendor | Handles single-period decisions |
| High service level requirements | (s,S) Policy | Better service level control |
| Cost minimization focus | EOQ or (s,Q) | Lower operational complexity |

### Parameter Sensitivity

**Most Sensitive Parameters:**
1. **Demand variability** - Higher variability requires higher safety stock
2. **Lead time uncertainty** - Longer/variable lead times increase costs
3. **Cost ratios** - Shortage vs. holding cost ratio drives policy choice
4. **Service level targets** - Higher targets require more inventory

**Less Sensitive Parameters:**
1. **Ordering costs** - Usually small impact on total cost
2. **Exact demand distribution** - Normal approximation often sufficient
3. **Minor parameter variations** - Models are generally robust

## Advanced Features

### Sensitivity Analysis

All models support sensitivity analysis to understand parameter impact:

```python
# Analyze impact of demand variability
demand_std_range = [2, 4, 6, 8, 10]
sensitivity = ss_policy.sensitivity_analysis('demand_std', demand_std_range)

# Plot results
import matplotlib.pyplot as plt
std_values = [r['parameter_value'] for r in sensitivity['results']]
costs = [r['total_cost'] for r in sensitivity['results']]
plt.plot(std_values, costs)
plt.xlabel('Demand Standard Deviation')
plt.ylabel('Total Cost')
plt.title('Cost Sensitivity to Demand Variability')
plt.show()
```

### Policy Comparison

Compare different inventory policies for the same scenario:

```python
# Compare (s,S) and (s,Q) policies
ss_result = ss_policy.run()
sq_result = sq_policy.run()

print("Policy Comparison:")
print(f"(s,S) total cost: ${ss_result.results['total_cost']:.2f}")
print(f"(s,Q) total cost: ${sq_result.results['total_cost']:.2f}")

if ss_result.results['total_cost'] < sq_result.results['total_cost']:
    savings = sq_result.results['total_cost'] - ss_result.results['total_cost']
    print(f"(s,S) policy saves ${savings:.2f} annually")
```

### Monte Carlo Analysis

For robust decision-making under uncertainty:

```python
# Run multiple scenarios with different random seeds
results = []
for seed in range(100):
    ss_policy.set_random_seed(seed)
    result = ss_policy.run()
    results.append(result.results['total_cost'])

# Analyze distribution of outcomes
import numpy as np
mean_cost = np.mean(results)
std_cost = np.std(results)
print(f"Expected cost: ${mean_cost:.2f} ± ${std_cost:.2f}")
```

## Performance Optimization

### Computational Efficiency

**For Large Simulations:**
- Use appropriate simulation length (365 days usually sufficient)
- Limit convergence tracking for very long simulations
- Consider parallel processing for sensitivity analysis

**Memory Management:**
- Large point datasets are only stored for small samples
- Convergence data can be disabled if not needed
- Use appropriate number of simulation runs

### Numerical Stability

**Common Issues and Solutions:**
- **Very small costs**: Scale parameters appropriately
- **Extreme parameter values**: Validate inputs before running
- **Convergence problems**: Check parameter reasonableness

## Troubleshooting

### Common Errors

1. **"Simulation not configured"**
   - Solution: Call `configure()` method or ensure proper initialization

2. **"Invalid parameters"**
   - Solution: Check parameter validation with `validate_parameters()`

3. **"Negative costs"**
   - Solution: Ensure all cost parameters are non-negative

4. **Poor service levels**
   - Solution: Increase reorder point or order-up-to level

### Validation Checklist

Before running simulations:
- [ ] All cost parameters are positive
- [ ] Demand parameters are reasonable
- [ ] Lead time parameters are non-negative
- [ ] Policy parameters make logical sense (s < S, Q > 0)
- [ ] Simulation length is appropriate for the scenario

## Best Practices

### Model Development
1. **Start simple**: Begin with EOQ for baseline understanding
2. **Add complexity gradually**: Move to stochastic models as needed
3. **Validate assumptions**: Check if model assumptions match reality
4. **Use real data**: Calibrate parameters with historical data

### Analysis Workflow
1. **Parameter estimation**: Use historical data to estimate parameters
2. **Model selection**: Choose appropriate model for the scenario
3. **Sensitivity analysis**: Understand parameter impact
4. **Policy comparison**: Compare alternative policies
5. **Implementation**: Monitor performance and adjust as needed

### Reporting Results
1. **Include confidence intervals**: Show uncertainty in results
2. **Document assumptions**: Clearly state model assumptions
3. **Provide recommendations**: Translate results into actionable insights
4. **Monitor performance**: Track actual vs. predicted performance

## Integration with Other Systems

### Data Sources
- **ERP systems**: Extract demand and cost data
- **Forecasting systems**: Use demand forecasts as inputs
- **Supplier systems**: Incorporate lead time data
- **Financial systems**: Validate cost parameters

### Output Integration
- **Planning systems**: Use optimal policies for planning
- **Monitoring dashboards**: Track key performance indicators
- **Reporting systems**: Generate regular performance reports
- **Decision support**: Provide what-if analysis capabilities

## References and Further Reading

### Academic References
- Silver, E. A., Pyke, D. F., & Peterson, R. (1998). Inventory Management and Production Planning and Scheduling
- Zipkin, P. H. (2000). Foundations of Inventory Management
- Porteus, E. L. (2002). Foundations of Stochastic Inventory Theory

### Practical Guides
- Muller, M. (2011). Essentials of Inventory Management
- Wild, T. (2017). Best Practice in Inventory Management
- Bragg, S. M. (2014). Inventory Management

### Online Resources
- INFORMS Practice Articles on Inventory Management
- Supply Chain Management Review
- Journal of Operations Management

---
