#! /usr/bin/python3
import numpy as np
import plotly.offline as plt
import plotly.graph_objs as go
import ast

cost_list = [1.00, 1.00, 1.00, 1.00, 7.27, 5.20, 15.50,
        102.90, 87.30, 87.30, 87.30, 100.90, 102.90]

# generate graphs from the data
total_positive = 34
total_negative = 86

with open("output.data", "r") as data_file:
    lines = data_file.read().splitlines()
    x_values = []
    y_values = []
    cost_values = []
    fn_rate_values = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,
            1.0,1.0,1.0,1.0,1.0,1.0]
    fp_rate_values = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,
            1.0,1.0,1.0,1.0,1.0,1.0]
    price_range_best_fscore = [0, 0, 0, 0, 0, 0, 0]
    price_range_best_cost = [0, 0, 0, 0, 0, 0, 0]
    price_range_best_fset = [[], [], [], [], [], [], []]
    for line in lines:
        line = ast.literal_eval(line)
        fn_rate = line[4][1][0] / total_positive # false negatives / t_p
        fp_rate = line[4][0][1] / total_negative # false positives / t_p
        cost = 0
        for elem in line[1]:
            cost += cost_list[elem]
        cost_values.append(cost)
        x_values.append(len(line[1])) # number of features
        y_values.append(line[2]) # fscore
        # now check if it can beat a feature set in its price range
        index = int((cost + 50) / 100)
        if price_range_best_fscore[index] < line[2]: 
            price_range_best_fscore[index] = line[2]
            price_range_best_fset[index] = line[1]
            price_range_best_cost[index] = cost
        elif price_range_best_fscore[index] == line[2]: 
            # replace the best if it costs more but has equal fscore
            # than the candidate
            if price_range_best_cost[index] > cost:
                price_range_best_fscore[index] = line[2]
                price_range_best_fset[index] = line[1]
                price_range_best_cost[index] = cost

        fn_index = len(line[1]) - 1
        if fn_rate_values[fn_index] > fn_rate:
            fn_rate_values[fn_index] = fn_rate
        if fp_rate_values[fn_index] > fp_rate:
            fp_rate_values[fn_index] = fp_rate

for i in range(0, 7):
    print(price_range_best_fset[i], price_range_best_cost[i],
            price_range_best_fscore[i])

trace_fscore = go.Scatter(
        x = x_values,
        y = y_values,
        mode = 'markers',
        )

layout_fscore = go.Layout(
        title = 'Performance graph',
        xaxis = dict(
            title = 'number of features',
            ),
        yaxis = dict(
            title = 'fscore',
            ),
        )

figure_fscore = go.Figure(data = [trace_fscore], layout = layout_fscore)
plt.plot(figure_fscore, filename='figure_fscore.html')

trace_cost = go.Scatter(
        x = cost_values,
        y = y_values,
        mode = 'markers',
        )

layout_cost = go.Layout(
        title = 'Performance graph',
        xaxis = dict(
            title = 'cost',
            ),
        yaxis = dict(
            title = 'fscore',
            ),
        )

figure_cost = go.Figure(data = [trace_cost], layout = layout_cost)
plt.plot(figure_cost, filename='figure_cost.html')

trace_fn_rate = go.Scatter(
        x = list(range(1, 14)),
        y = fn_rate_values,
        mode = 'lines+markers',
        name = 'false negative rate',
        line = dict(
            shape = "spline"
            )
        )

trace_fp_rate = go.Scatter(
        x = list(range(1, 14)),
        y = fp_rate_values,
        mode = 'lines+markers',
        name = 'false positive rate',
        line = dict(
            shape = "spline"
            )
        )

layout_fn_rate = go.Layout(
        title = 'Performance graph',
        xaxis = dict(
            title = 'feature_numbers',
            ),
        yaxis = dict(
            title = 'rate',
            ),
        )

figure_fn_rate = go.Figure(data = [trace_fn_rate, trace_fp_rate], layout = layout_fn_rate)
plt.plot(figure_fn_rate, filename='figure_fn_rate.html')
