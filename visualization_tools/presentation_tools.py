"""Contains all classes to handle the automation of visualizations sent to Google Slides.

It will primarily use the gslides package, but will eventually upload images to Google Slides.

Class ideas:

    - AutomatedPresentation
    - AutomatedSlide
    - VisualizationAutomator

Necessary objects and methods:

    - Google Slides connection
    - Google Sheets connection
    - Data aggregation for Google Charts
        - Automatic column type detection
    - Automatic checking for slide deck and updating if necessary.

I'd like to be able to automatically create a slide-deck and automatically update it with new
or modified visualizations. I'd like it to detect whether or not to create a histogram or a 
column chart depending on if the column contains integers/floats or strings.

I'd also like to be able to update an existing slide show. I would like to either update
all of the slides in the slideshow 


"""
import pandas as pd
from gslides import * 
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.cloud import storage    
import datetime
import gslides
import os
from typing import Optional, List
from gslides import (
    Frame,
    Presentation,
    Spreadsheet,
    Table,
    Series, Chart
)



class ImagePresentation:
    """
    This class takes a list of image file paths, uploads those images to Google Cloud Platform (GCP),
    obtains signed URLs for each image on GCP, and then uses those signed URLs to create a Google Slides
    presentation with each image on a different slide.

    Attributes:
        image_paths (list): A list of file paths to the images.
        presentation_name (str): The name of the Google Slides presentation to be created.
        project_id (str, optional): The GCP project ID. Defaults to "empirical-weft-349320".
        gcp_bucket_name (str, optional): The GCP storage bucket name. Defaults to "gslide_images".
    """
    def __init__(self, 
                image_paths, 
                presentation_name, 
                project_id="empirical-weft-349320",
                gcp_bucket_name="gslide_images"):
        self.GSLIDES_SCOPES = ["https://www.googleapis.com/auth/presentations"]
        self.image_paths = image_paths
        self.presentation_name = presentation_name
        self.presentation_id = None
        self.project_id = project_id
        self.gcp_storage_client = None
        self.gcp_bucket_name = gcp_bucket_name
        self.google_slides_service = None
        self.creds = None

        self._connect_to_gcp()
        self._connect_to_google_slides()
        self.pres = Presentation.create(name = self.presentation_name)
        self.presentation_id = self.pres.presentation_id

    def upload_images_to_slides(self,
                                magnitude=6000000,
                                scaleX=1,
                                scaleY=1,
                                translateX=0,
                                translateY=0):
        """
        Upload images from the given file paths to the Google Slides presentation.
        
        Args:
            magnitude (int, optional): The size of the image in EMU (English Metric Units). Defaults to 6000000.
            scaleX (float, optional): The horizontal scaling factor of the image. Defaults to 1.
            scaleY (float, optional): The vertical scaling factor of the image. Defaults to 1.
            translateX (int, optional): The horizontal translation of the image in EMU. Defaults to 0.
            translateY (int, optional): The vertical translation of the image in EMU. Defaults to 0.
        """
        for image_path in self.image_paths:
            url = self.upload_image_to_gcp(image_path)
            slide_id = self.add_slide()
            self.upload_image_to_slide(url, 
                                       slide_id,
                                       scaleX=scaleX,
                                       scaleY=scaleY,
                                       translateX=translateX,
                                       translateY=translateY)

    def upload_image_to_slide(self, 
                              url, 
                              slide_id,
                              magnitude=6000000,
                              scaleX=1,
                              scaleY=1,
                              translateX=0,
                              translateY=0):
        """
        Upload an image to a specified slide using the image's signed URL.
        
        Args:
            url (str): The signed URL of the image on GCP.
            slide_id (str): The ID of the slide to add the image to.
            magnitude (int, optional): The size of the image in EMU (English Metric Units). Defaults to 6000000.
            scaleX (float, optional): The horizontal scaling factor of the image. Defaults to 1.
            scaleY (float, optional): The vertical scaling factor of the image. Defaults to 1.
            translateX (int, optional): The horizontal translation of the image in EMU. Defaults to 0.
            translateY (int, optional): The vertical translation of the image in EMU. Defaults to 0.

        Returns:
            str: The ID of the created image on the slide.
        """

        # Create a new image, using the supplied object ID,
        # with content downloaded from IMAGE_URL.
        IMAGE_URL = url
        requests = []
        emu4M = {
            'magnitude': magnitude,
            'unit': 'EMU'
        }
        requests.append({
            'createImage': {
                'url': IMAGE_URL,
                'elementProperties': {
                    'pageObjectId': slide_id,
                    'size': {
                        'height': emu4M,
                        'width': emu4M
                    }
                    ,
                    'transform': {
                        'scaleX': scaleX,
                        'scaleY': scaleY,
                        'translateX': translateX,
                        'translateY': translateY,
                        'unit': 'EMU'
                    }
                }
            }
        })

        # Execute the request.
        body = {
            'requests': requests
        }
        response = (self.google_slides_service
                        .presentations()
                        .batchUpdate(presentationId=self.pres.presentation_id, body=body)
                        .execute()
                )
        create_image_response = response.get('replies')[0].get('createImage')
        image_id = create_image_response.get('objectId')
        return image_id

    def upload_image_to_gcp(self, image_path):
        """
        Upload an image to Google Cloud Platform (GCP) and generate a signed URL for the image.
        
        Args:
            image_path (str): The file path to the image.

        Returns:
            str: The signed URL of the uploaded image on GCP.
        """
        blob_name = image_path.split("/")[-1] + "-"
        blob = self.gcp_storage_client.bucket(self.gcp_bucket_name).blob(blob_name)
        blob.upload_from_filename(image_path)

        url = blob.generate_signed_url(
                        version="v4",
                        # This URL is valid for 15 minutes
                        expiration=datetime.timedelta(minutes=15),
                        # Allow GET requests using this URL.
                        method="GET"
                    )
        return url

    def _connect_to_gcp(self):
        """Establish a connection to Google Cloud Platform.

        This method connects to Google Cloud Platform (GCP) using the credentials
        provided in the gslides package. It initializes the GCP storage client for
        later use in uploading images to Google Cloud Storage.

        Images are uploaded to Google Cloud Storage (GCS) and then added to the
        Google Slides presentation via a signed URL.
        """
        service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        storage_client = (storage.Client()
                            .from_service_account_json(service_account_path, 
                                                       project=self.project_id))
        self.gcp_storage_client = storage_client

    def _connect_to_google_slides(self):
        """Establish a connection to the Google Slides API.

        This method connects to the Google Slides API using the credentials stored
        in 'token.json' file. It initializes the google_slides_service attribute
        for later use in creating and manipulating presentations.
        """
        creds = Credentials.from_authorized_user_file('token.json', self.GSLIDES_SCOPES)
        gslides.initialize_credentials(creds) #BringYourOwnCredentials
        service = build('slides', 'v1', credentials=creds)
        self.google_slides_service = service
        self.creds = creds

    def add_slide(self):
        """Add a new slide to the presentation and return its ID.

        This method adds a new slide to the Google Slides presentation using
        the 'BLANK' predefined layout. The method returns the ID of the
        newly created slide, which can be used for further manipulations.

        Returns:
            str: The ID of the newly created slide.
        """
        # Add a slide at index 1 using the predefined
        # 'TITLE_AND_TWO_COLUMNS' layout and the ID page_id.
        requests = [
            {
                'createSlide': {
                    'slideLayoutReference': {
                        'predefinedLayout': 'BLANK'
                    }
                }
            }
        ]

        # If you wish to populate the slide with elements,
        # add element create requests here, using the page_id.

        # Execute the request.
        body = {
            'requests': requests
        }
        response = self.google_slides_service.presentations() \
            .batchUpdate(presentationId=self.presentation_id, body=body).execute()
        create_slide_response = response.get('replies')[0].get('createSlide')
        print('Created slide with ID: {0}'.format(
            create_slide_response.get('objectId')))
        page_id = create_slide_response.get('objectId')
        return page_id

class HistogramPresentation:
    """This is class which takes a dataframe of columns which are meant to be put into a histogram
    or a column chart.
    
    input_df: pandas.DataFrame with columns to be converted to a series of Histograms in Google Slides.
    """
    def __init__(self, input_df: pd.DataFrame, presentation_id=None):
        """Initializes HistogramPresentation object."""
        self.input_df = input_df
        self.scopes = ['https://www.googleapis.com/auth/presentations',
                       'https://www.googleapis.com/auth/spreadsheets']
        self.creds = None
        self.presentation_id = presentation_id

    def create(self, 
               presentation_name, 
               target_columns: Optional[List[str]] = None,
               replace_slides: bool = True):
        """Create a Google Slides presentation with a histogram or column chart for each column.

        Args:
            presentation_name (str): The name of the new Google Slides presentation.
            target_columns (list, optional): A list of column names to be plotted. If not provided,
                all columns in the input DataFrame will be plotted.
            replace_slides (bool, optional): If True, existing slides in the presentation will be
                replaced. Default is True.
        """
        self._connect_to_google()
        # Check that the format of target_columns are correct.
        if target_columns and not isinstance(target_columns, list):
            raise Exception("target_columns must be a list")
        sheet_names = ["main_df"] + list(self.input_df.columns)

        if self.presentation_id:
            pres = Presentation.get(self.presentation_id)
            # Delete all old slides to add new ones.
            if replace_slides:
                for slide_id in pres.slide_ids:
                    pres.rm_slide(slide_id)
        else:
            pres = Presentation.create(name = presentation_name)
            self.presentation_id = pres.presentation_id
        
        spr = Spreadsheet.create(
            title = f"Data for Presentation: {presentation_name}",
            sheet_names=sheet_names
        )

        main_dataframe = Frame.create(df = self.input_df,
                                spreadsheet_id=spr.spreadsheet_id,
                                sheet_id=spr.sheet_names[sheet_names[0]],
                                sheet_name=sheet_names[0],
                                overwrite_data=True)
        
        for idx, column in enumerate(self.input_df.columns):
            print(f"The column: \n\n{column}\n\nis being processed.\n")
            formatted_df, col_type_is_string = self._reformat_column(target_column=column)
            frame = Frame.create(df = formatted_df,
                                spreadsheet_id=spr.spreadsheet_id,
                                sheet_id=spr.sheet_names[sheet_names[idx+1]],
                                sheet_name=sheet_names[idx+1],
                                overwrite_data=True)

            # Create a histogram for integers and floats, column for strings
            if col_type_is_string:
                chart = Series.column(series_columns=[column])
            else:
                chart = Series.histogram(series_columns=[column], bucket_size = 10)

            # We don't add the 
            ch = Chart(
                data = frame.data,
                x_axis_column = "Response",
                series = [chart],
                title = " ",
                x_axis_label = "Response",
                y_axis_label = 'Count',
                legend_position="NO_LEGEND",
            )

            pres.add_slide(objects=[ch],
                            layout=(1,1),
                            title=column,
                            notes=" ")

            pres.show_slide(slide_id=pres.slide_ids[-1])

    def _connect_to_google(self):
        """Connect to Google using the credentials in 'token.json'."""
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', self.scopes)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    './visualization_tools/token.json', self.scopes)
                creds = flow.run_local_server()
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        gslides.initialize_credentials(creds) #BringYourOwnCredentials
        self.creds = creds

    def _reformat_column(self, 
                         target_column: str):
        """Reformat the target column if its values are strings.

        Args:
            target_column (str): The name of the target column.

        Returns:
            pd.DataFrame: A reformatted DataFrame with the target column.
            bool: True if the target column contains strings, False otherwise.
        """
        # Check to make sure the values in the target column aren't a string
        try:
            self.input_df[target_column].astype(float)
            return self.input_df, False
        except ValueError:
            return (pd.DataFrame(self.input_df[target_column].value_counts())
                    .reset_index()
                    .rename(columns={"index": "Response"})
                    .sort_values(by=["Response"])), True

if __name__ == "__main__":
    #%env GOOGLE_APPLICATION_CREDENTIALS /Users/wthompson/Documents/wjtho/ContractWork/similarity-matrix/empirical-weft-349320-037d5589f437.json

    folder = "../image_extraction/montera_topic_extractions"

    image_names = os.listdir("./image_extraction/montera_topic_extractions")
    image_paths = [folder+"/"+image_name for image_name in image_names][0:6]

    image_presentation = ImagePresentation(image_paths, "Demo Presentation")
    image_presentation.upload_images_to_slides(magnitude=6000000*3,
                                                scaleX=1.5,
                                                scaleY=1.5,
                                                translateX=80000,
                                                translateY=-2000000)