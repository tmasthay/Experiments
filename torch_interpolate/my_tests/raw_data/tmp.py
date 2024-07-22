
    if 'name' not in cfg.keys():
        raise ValueError('Config must contain a name field')

    def lambdafy(expr):
        return eval(f'lambda {cfg.function.variable_name}: {expr}')

    # Read in the function and its derivative from the config
    function = lambdafy(cfg.function.formula)
    derivative = lambdafy(cfg.function.derivative)

    # Generate points and compute function values
    points = torch.linspace(cfg.points.start, cfg.points.end, cfg.points.num)
    values = function(points)

    # Create the interpolator
    gi = torch_interpolations.RegularGridInterpolator([points], values)

    # Define points to interpolate and compute interpolated values
    interp_points = torch.linspace(
        cfg.points.start, cfg.points.end, cfg.points.num, requires_grad=True
    )
    interp_values = gi([interp_points])

    # Compute the autodiff derivative for interpolated function
    interp_values.sum().backward()
    autodiff_derivative = interp_points.grad

    # Compute the analytical derivative for comparison
    analytical_derivative = derivative(interp_points)

    # Compute the finite difference derivative for interpolated function
    h = 1e-3
    interp_points_fd = interp_points.detach().clone()
    interp_points_fd.requires_grad = False
    interp_values_plus_h = gi([interp_points_fd + h])
    interp_values_minus_h = gi([interp_points_fd - h])
    finite_difference_derivative = (
        interp_values_plus_h - interp_values_minus_h
    ) / (2 * h)

    # Compute the function values and derivatives directly (without interpolation)
    points.requires_grad = True
    function_values = function(points)
    function_values.sum().backward()
    vanilla_autodiff_derivative = points.grad

    # Compute the finite difference derivative for the original function
    points_fd = points.detach().clone()
    points_fd.requires_grad = False
    function_values_plus_h = function(points_fd + h)
    function_values_minus_h = function(points_fd - h)
    vanilla_finite_difference_derivative = (
        function_values_plus_h - function_values_minus_h
    ) / (2 * h)

    # Plot the results
    plt.figure(figsize=(18, 12))

    order = [1, 3, 5, 2, 4, 6]
    shape = [3, 2]

    # Plot the interpolated function
    plt.subplot(*shape, order[0])
    plt.plot(
        interp_points.detach().numpy(),
        interp_values.detach().numpy(),
        'r',
        label='Interpolated Function',
        alpha=0.5,
    )
    plt.plot(
        points.detach().numpy(),
        values.detach().numpy(),
        'bo',
        label='Original Points',
        markersize=2,
    )
    plt.title('Interpolated Function')
    plt.xlabel('X')
    plt.ylabel('Function Value')
    plt.legend()
    plt.grid(True)

    # Plot the autodiff and analytical derivatives for the interpolated function
    plt.subplot(*shape, order[1])
    plt.plot(
        interp_points.detach().numpy(),
        analytical_derivative.detach().numpy(),
        'r',
        label='Analytical Derivative',
        alpha=0.5,
    )
    plt.plot(
        interp_points.detach().numpy(),
        autodiff_derivative.detach().numpy(),
        'b--',
        label='Autodiff Derivative',
    )
    plt.title('Interpolated Derivative Comparison: Autodiff vs Analytical')
    plt.xlabel('X')
    plt.ylabel('Derivative Value')
    plt.legend()
    plt.grid(True)

    # Plot the finite difference derivative for the interpolated function
    plt.subplot(*shape, order[2])
    plt.plot(
        interp_points.detach().numpy(),
        analytical_derivative.detach().numpy(),
        'r',
        label='Analytical Derivative',
        alpha=0.5,
    )
    plt.plot(
        interp_points.detach().numpy(),
        finite_difference_derivative.detach().numpy(),
        'b--',
        label='Finite Difference Derivative',
    )
    plt.title(
        'Interpolated Derivative Comparison: Finite Difference vs Analytical'
    )
    plt.xlabel('X')
    plt.ylabel('Derivative Value')
    plt.legend()
    plt.grid(True)

    # Plot the original function
    plt.subplot(*shape, order[3])
    plt.plot(
        points.detach().numpy(),
        function_values.detach().numpy(),
        'r',
        label='Original Function',
        alpha=0.5,
    )
    plt.plot(
        points.detach().numpy(),
        values.detach().numpy(),
        'bo',
        label='Original Points',
        markersize=2,
    )
    plt.title('Original Function')
    plt.xlabel('X')
    plt.ylabel('Function Value')
    plt.legend()
    plt.grid(True)

    # Plot the autodiff and analytical derivatives for the original function
    plt.subplot(*shape, order[4])
    plt.plot(
        points.detach().numpy(),
        derivative(points).detach().numpy(),
        'r',
        label='Analytical Derivative',
        alpha=0.5,
    )
    plt.plot(
        points.detach().numpy(),
        vanilla_autodiff_derivative.detach().numpy(),
        'b--',
        label='Vanilla Autodiff Derivative',
    )
    plt.title('Original Derivative Comparison: Autodiff vs Analytical')
    plt.xlabel('X')
    plt.ylabel('Derivative Value')
    plt.legend()
    plt.grid(True)

    # Plot the finite difference derivative for the original function
    plt.subplot(*shape, order[5])
    plt.plot(
        points.detach().numpy(),
        derivative(points).detach().numpy(),
        'r',
        label='Analytical Derivative',
        alpha=0.5,
    )
    plt.plot(
        points.detach().numpy(),
        vanilla_finite_difference_derivative.detach().numpy(),
        'b--',
        label='Vanilla Finite Difference Derivative',
    )
    plt.title('Original Derivative Comparison: Finite Difference vs Analytical')
    plt.xlabel('X')
    plt.ylabel('Derivative Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    fig_name = f'{cfg.name}_{cfg.function.freq}.{cfg.output.format}'
    fig_name = hydra_out(fig_name)
    plt.savefig(fig_name)

    print(f'Figure saved to {fig_name}')

    plt.clf()
