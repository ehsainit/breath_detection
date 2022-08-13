from thermal_image_processing import ThermalDataAnalyzer
from depth_image_processing import DepthDataAnalyzer
from video_analysis.utils import generate_plots_from_data

if __name__ == '__main__':
    project_path = "C:\\Users\\t-bomb\\Desktop\\fadi\\breath_detection\\record_data\\ceac5803-85ff-4f0a-803e-20938c96d28c"
    thermo_analyzer = ThermalDataAnalyzer(project_path)
    depth_analyzer = DepthDataAnalyzer(project_path)
    thermo_analysis_data= thermo_analyzer.analyze()
    generate_plots_from_data(thermo_analysis_data, project_path, "thermal")
    depth_analysis_data = depth_analyzer.analyze()
    generate_plots_from_data(depth_analysis_data, project_path, "depth")
