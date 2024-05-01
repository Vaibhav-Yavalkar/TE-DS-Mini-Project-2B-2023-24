from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="app-home"),
    path('main/', views.main, name="app-upload"),
    path('upload/', views.upload_audio, name="upload"),
    path('about/', views.about, name="about"),
]