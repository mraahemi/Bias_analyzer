$(document).ready(function() {
	$(chart_id).highcharts({
		chart: {renderTo: 'chart_ID',
                type: 'bar', height: 1500, length: 1000},
		title: {text: 'Importance of Attributes'},
		xAxis: {
		categories: feature_cols // ['capital-gain', 'capital-loss', ...]
		},
		yAxis: {title:
		{"text": 'yAxis Label'}},
		series: [{"name": 'Label1', "data": shap_values}] // shap_values = [-1.82,
		// {"capital-gain": -1.82, "capital-loss":
	});
});
