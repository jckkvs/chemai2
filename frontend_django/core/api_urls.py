"""core/api_urls.py - API URLルーティング"""
from django.urls import path
from . import views

urlpatterns = [
    path("session/<uuid:session_id>/upload/", views.upload_data, name="api_upload"),
    path("session/<uuid:session_id>/columns/", views.set_columns, name="api_set_columns"),
    path("session/<uuid:session_id>/sample/", views.load_sample, name="api_load_sample"),
    path("session/<uuid:session_id>/descriptors/", views.calculate_descriptors, name="api_calculate_descriptors"),
    # パラメータ自動UI用API
    path("params/model/<str:model_key>/", views.get_model_params_schema, name="api_model_params"),
    path("params/adapter/<str:adapter_name>/", views.get_adapter_params_schema, name="api_adapter_params"),
]

