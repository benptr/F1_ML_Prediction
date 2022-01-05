from django.apps import AppConfig

class MlApplicationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ML_Application'
    def ready(self):
        from Backend import Viz as viz
        viz.init_viz()
        import Backend.predictions as pred
        pred.init_from_local()
            
