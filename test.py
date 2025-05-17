from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import sqlite3
import PyPDF2
from PIL import Image
import pytesseract
import spacy
import aiohttp
import re
from datetime import datetime

print(os.getenv("GOOGLE_API_KEY"))