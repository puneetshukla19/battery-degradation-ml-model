from django.urls import path
from . import views

urlpatterns = [
    path("",                           views.dashboard_page,         name="dashboard"),
    path("executive/",                 views.executive_summary_page, name="executive-summary"),
    path("executive",                  views.executive_summary_page, name="executive-summary-noslash"),
    path("api/overview/",              views.api_overview,      name="api-overview"),
    path("api/fleet-trend/",           views.api_fleet_trend,   name="api-fleet-trend"),
    path("api/quintiles/",             views.api_quintiles,     name="api-quintiles"),
    path("api/vehicles/",              views.api_vehicles,      name="api-vehicles"),
    path("api/bayes-coef/",            views.api_bayes_coef,    name="api-bayes-coef"),
    path("api/bayes-coef/<str:reg>/",  views.api_bayes_coef,    name="api-bayes-coef-veh"),
    path("api/soh-scatter/",              views.api_soh_scatter,       name="api-soh-scatter"),
    path("api/soh-delta-trend/",          views.api_soh_delta_trend,   name="api-soh-delta-trend"),
    path("api/efc-trend/",               views.api_efc_trend,         name="api-efc-trend"),
    path("api/anomaly-tiers/",         views.api_anomaly_tiers,      name="api-anomaly-tiers"),
    path("api/anomaly-breakdown/",              views.api_anomaly_breakdown, name="api-anomaly-breakdown"),
    path("api/anomaly-breakdown/<str:reg>/",    views.api_anomaly_breakdown, name="api-anomaly-breakdown-veh"),
    path("api/soh-bands/<str:reg>/",            views.api_soh_bands,         name="api-soh-bands"),
    path("api/sessions/<str:reg>/",    views.api_sessions,            name="api-sessions"),
    path("api/telemetry/<str:reg>/<str:session_id>/", views.api_telemetry, name="api-telemetry"),
    path("api/rul-timeline/<str:reg>/",  views.api_rul_timeline,      name="api-rul-timeline"),
    path("api/breakdown-timeline/", views.api_breakdown_timeline, name="api-breakdown-timeline"),
    path("api/distributions/",      views.api_distributions,      name="api-distributions"),
]
