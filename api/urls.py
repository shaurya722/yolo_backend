
from django.urls import path
from . import views

urlpatterns = [
    path('bike-detection/', views.YoloPredict.as_view()),  
    path('bike-video-detection/', views.YoloPredictBikeDetectionVideo.as_view()),  
]
