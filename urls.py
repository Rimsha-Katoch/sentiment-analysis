from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('results/', views.results, name='results'),
    path('analytics/', views.analytics, name='analytics'),
    path('clear/', views.clear_session, name='clear_session'),
    path('delete-last-50/', views.delete_last_50_records, name='delete_last_50_records'),
]
