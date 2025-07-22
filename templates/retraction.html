<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Retraction Status</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        .status-card {
            margin-bottom: 2rem;
        }
        .status-list {
            list-style: none;
            padding-left: 0;
        }
        .status-list li {
            margin-bottom: 1rem;
        }
        .horizontal-list {
            list-style: none;
            padding-left: 0;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .horizontal-list li {
            background: #f1f1f1;
            padding: 5px 10px;
            border-radius: 5px;
        }
        .oa-available a {
            font-weight: bold;
        }
        .table-responsive {
            margin-top: 2rem;
        }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h2 class="mb-4"><strong>Retraction Status for : {{ doi }} </strong></h2>
        
        <!-- Combined Status Card -->
        <div class="card status-card {% if retracted %}retracted-true{% else %}retracted-false{% endif %}">
            <div class="card-header {% if retracted %}bg-danger{% else %}bg-success{% endif %} text-white">
                <h5 class="mb-0">DOI: {{ doi }}</h5>
            </div>
            <div class="card-body">
                <ul class="status-list">
                    <!-- Open Access Status -->
                    <li>
                        {% if oa[0] == 'Yes' %}
                        <strong>You can download the full text of this paper here: </strong><br>
                        <span class="oa-available">
                            <center><a href="{{ oa[1] }}" target="_blank">{{ oa[1] }}</a></center><br>
                        </span>
                        {% else %}
                        <strong>The full text of this paper is not available.</strong><br>
                        {% endif %}
                    </li>

                    <!-- Retraction Status -->
                    <li>
                        <strong>Retraction Status:</strong>
                        <span class="badge status-badge {% if retracted %}bg-danger{% else %}bg-success{% endif %}">
                            {% if retracted %}Retracted{% else %}Not Retracted{% endif %}
                        </span>
                        {% if retracted %}
                        <div class="mt-2 text-muted">
                            <strong>Reasons:</strong> 
                            <ul class="horizontal-list">
                                {% for reason in reasons %}
                                    <li>{{ reason }}</li>
                                {% endfor %}
                            </ul>
                        </div>
                        {% endif %}
                    </li>
                </ul>
            </div>
        </div>

        <!-- Results Table -->
        {% if resu is not none and resu|length > 0 %}
        <div class="table-responsive">
            <h4 class="mt-4">Citing Papers Also Retracted</h4>
            <table class="table table-bordered table-striped">
                <thead class="thead-dark">
                    <tr>
                        {% for col in resu_column_names %}
                        <th>{{ col }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for row in resu %}
                    <tr>
                        {% for col in resu_column_names %}
                        <td>{{ row[col] }}</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% else %}
        <div class="alert alert-info mt-4">No citing papers found that have been retracted.</div>
        {% endif %}
    </div>
</body>
</html>
