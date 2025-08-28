import matplotlib.pyplot as plt

def plot_boxplots(logs, models):
    properties = logs.columns.get_level_values('property').unique()
    metrics = logs.columns.get_level_values('metric').unique()

    fig, axs = plt.subplots(
        len(properties), 1, 
        figsize=(max(len(models), 4), 4), 
        sharex=True
    )

    for row, property in enumerate(properties):
        axs[row].set_title(property)
        axs[row].set_ylabel('MAE (Hartree)')

        tick_labels, data = [], []
        for label, model_type, radial_function_type, architecture in models:
            tick_labels.append(label)
            ensemble = logs[model_type, radial_function_type, architecture, 'val', property, 'mae'].dropna()
            ensemble_means = ensemble.min()
            data.append(ensemble_means.to_list())

        axs[row].boxplot(data, tick_labels=tick_labels)
    
    fig.tight_layout()

def print_latex_table(logs, models):
    properties = logs.columns.get_level_values('property').unique()

    print('\\hline')
    print('& ', ' & '.join(properties), '\\\\')
    print('\\hline')

    for label, model_type, radial_function_type, architecture in models:
        row_str = [label]
        for property in properties:

            ensemble = logs[model_type, radial_function_type, architecture, 'val', property, 'mae']
            ensemble_mins = ensemble.min()

            mu = ensemble_mins.mean().item() * 27.2114
            std = ensemble_mins.std().item() * 27.2114

            row_str.append(f'${mu:.4f} \pm {std:.4f}$')

        print(' & '.join(row_str), '\\\\')
        print('\\hline')