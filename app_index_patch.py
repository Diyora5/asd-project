from flask import send_from_directory

def register_index_route(app):
    @app.route('/')
    def index():
        return send_from_directory('static', 'asd_project_mobile_app.html')
