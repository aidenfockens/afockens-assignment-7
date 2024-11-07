from flask import Flask, render_template, request, url_for, jsonify, session
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
import uuid
from scipy import stats

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  

def generate_plots(N, mu, sigma2, S, beta0, beta1):
    plot1_filename = f"plot1_{uuid.uuid4().hex}.png"
    plot2_filename = f"plot2_{uuid.uuid4().hex}.png"
    plot1_path = f"./static/{plot1_filename}"
    plot2_path = f"./static/{plot2_filename}"

    # Clear old plot files in the static folder
    for filename in os.listdir("./static"):
        if filename.startswith("plot1_") or filename.startswith("plot2_"):
            os.remove(os.path.join("./static", filename))

    X = np.random.rand(N)
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Plot 1: Scatter with regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', label='Data')
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red', label=f'Y = {slope:.2f} * X + {intercept:.2f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot with Regression Line')
    plt.legend()
    plt.savefig(plot1_path)
    plt.close()

    # Simulation for slopes and intercepts
    slopes, intercepts = [], []
    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    session['simulated_slopes'] = slopes
    session['simulated_intercepts'] = intercepts

    # Plot 2: Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot2_path)
    plt.close()

    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S
    session['N'] = N
    session['S'] = S
    session['slope'] = slope
    session['intercept'] = intercept
    session['beta0'] = beta0
    session['beta1'] = beta1
    return plot1_filename, plot2_filename, slope_more_extreme, intercept_more_extreme


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        plot1_filename, plot2_filename, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S, beta0, beta1)

        response_data = {
            "plot1_url": url_for("static", filename=plot1_filename),
            "plot2_url": url_for("static", filename=plot2_filename),
            "slope_extreme": slope_extreme,
            "intercept_extreme": intercept_extreme
        }

        # Check if the request is AJAX and return JSON if so
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify(response_data)

        # If not an AJAX request, render the template with the response data
        return render_template("index.html", **response_data)

    # Handle GET request by rendering the main page
    return render_template("index.html")


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    N = session.get("N")
    S = session.get("S")
    slope = session.get("slope")
    intercept = session.get("intercept")
    beta0 = session.get("beta0")
    beta1 = session.get("beta1")
    slopes = session.get("simulated_slopes", [])
    intercepts = session.get("simulated_intercepts", [])

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    simulated_stats = slopes if parameter == "slope" else intercepts
    observed_stat = slope if parameter == "slope" else intercept
    hypothesized_value = beta1 if parameter == "slope" else beta0

    # Calculate p-value based on test type
    if test_type == ">":
        p_value = sum(1 for v in simulated_stats if v >= observed_stat) / len(simulated_stats)
    elif test_type == "<":
        p_value = sum(1 for v in simulated_stats if v <= observed_stat) / len(simulated_stats)
    else:
        p_value = sum(1 for v in simulated_stats if abs(v - observed_stat) >= abs(observed_stat)) / len(simulated_stats)

    fun_message = "Rare event! You encountered an extremely small p-value." if p_value <= 0.0001 else ""

    plot3_filename = f"plot3_{uuid.uuid4().hex}.png"
    plot3_path = f"./static/{plot3_filename}"
    for filename in os.listdir("./static"):
        if filename.startswith("plot3_"):
            os.remove(os.path.join("./static", filename))


    plt.figure(figsize=(8, 6))
    plt.hist(simulated_stats, bins=20, alpha=0.7, color="gray")
    plt.axvline(observed_stat, color="blue", linestyle="--", label=f'Observed {parameter}: {observed_stat:.2f}')
    plt.axvline(hypothesized_value, color="red", linestyle="--", label=f'Hypothesized {parameter}: {hypothesized_value:.2f}')
    plt.legend()
    plt.title("Histogram of Simulated Statistics for Hypothesis Testing")
    plt.savefig(plot3_path)
    plt.close()

    # Ensure the route always returns JSON
    return jsonify({
        "p_value": p_value,
        "fun_message": fun_message,
        "plot3_url": url_for('static', filename=plot3_filename)
    })

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))
    beta0 = session.get("beta0")
    beta1 = session.get("beta1")
   



    estimates = np.array(session['simulated_slopes']) if parameter == "slope" else np.array(session['simulated_intercepts'])
    mean_estimate = np.mean(estimates)
    std_error = np.std(estimates, ddof=1) / np.sqrt(len(estimates))

    t_critical = stats.t.ppf((1 + confidence_level) / 2, len(estimates) - 1)
    margin_of_error = t_critical * std_error
    ci_lower, ci_upper = mean_estimate - margin_of_error, mean_estimate + margin_of_error
    true_param = beta1 if parameter == "slope" else beta0

    plot4_filename = f"plot4_{uuid.uuid4().hex}.png"
    plot4_path = f"./static/{plot4_filename}"

    for filename in os.listdir("./static"):
        if filename.startswith("plot4_"):
            os.remove(os.path.join("./static", filename))


    plt.figure(figsize=(10, 6))
    plt.hist(estimates, bins=30, color="gray", alpha=0.7)
    plt.axvline(mean_estimate, color="blue", linestyle="--", label=f"Mean {parameter}: {mean_estimate:.2f}")
    plt.axvline(ci_lower, color="green", linestyle="--", label=f"CI Lower: {ci_lower:.2f}")
    plt.axvline(ci_upper, color="green", linestyle="--", label=f"CI Upper: {ci_upper:.2f}")
    plt.xlabel(f"{parameter.capitalize()} Estimate")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title(f"Confidence Interval for {parameter.capitalize()} Estimate")
    plt.savefig(plot4_path)
    plt.close()

    return jsonify({"ci_lower": ci_lower, 
                    "ci_upper": ci_upper, 
                    "includes_true": true_param, 
                    "plot4_url": url_for('static', filename=plot4_filename)})


if __name__ == "__main__":
    app.run(debug=True)
