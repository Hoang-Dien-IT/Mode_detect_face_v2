import os
import cv2
import numpy as np
from scipy.spatial.distance import cosine
import time
import threading
import tensorflow as tf
from retinaface import RetinaFace
from deepface import DeepFace
import traceback
import concurrent.futures
from collections import OrderedDict
import pickle
import joblib
import multiprocessing

# Cố gắng import các thư viện tối ưu
try:
    import faiss
    FAISS_AVAILABLE = True
    print("FAISS được tìm thấy: Sẽ sử dụng tìm kiếm vector hiệu suất cao")
except ImportError:
    FAISS_AVAILABLE = False
    try:
        from annoy import AnnoyIndex
        ANNOY_AVAILABLE = True
        print("Annoy được tìm thấy: Sẽ sử dụng tìm kiếm gần đúng Annoy")
    except ImportError:
        ANNOY_AVAILABLE = False
        print("Không tìm thấy FAISS hoặc Annoy: Sẽ sử dụng tìm kiếm tuần tự")

# Cấu hình GPU nâng cao
print("Đang cấu hình GPU...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Thiết lập sử dụng GPU đầu tiên nếu có nhiều GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        # Thiết lập chế độ tăng trưởng bộ nhớ thích ứng
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Thiết lập cụ thể mức sử dụng bộ nhớ GPU để tối ưu hiệu năng
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
        )
        print(f"Đã tìm thấy và cấu hình GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"Lỗi khi cấu hình GPU: {e}")
else:
    print("Không tìm thấy GPU, sẽ sử dụng CPU")

# Đường dẫn thư mục
KNOWN_FACES_DIR = r"d:\Luan_van\model_2\known_faces"
EMBEDDINGS_DIR = r"d:\Luan_van\model_2\data\embeddings"
TEMP_DIR = r"d:\Luan_van\model_2\temp"

# Tạo thư mục temp nếu chưa tồn tại
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Khởi tạo biến global
latest_recognition = []  # Khởi tạo biến global để lưu kết quả nhận dạng gần nhất
processing = False
# Không pre-load model để tiết kiệm bộ nhớ, sẽ được tải khi cần thiết
arcface_model = None

# Tham số tối ưu hóa hiệu suất
# Định nghĩa các tham số tối ưu để triển khai thực tế
THRESHOLD = 0.5  # Tăng ngưỡng lên để giảm độ chặt chẽ trong so sánh khuôn mặt
DETECTION_INTERVAL = 1  # Xử lý mỗi frame để tăng độ mượt
FACE_CACHE_SIZE = 200
RECOGNITION_BUFFER_SIZE = 8
MAX_FACES = 3  # Có thể nhận diện đồng thời 3 khuôn mặt
USE_LIGHTWEIGHT_MODEL = False  # Sử dụng ArcFace cho độ chính xác cao hơn
DETECTION_SIZE = 240  # Kích thước frame phát hiện khuôn mặt tối ưu
DISPLAY_WIDTH = 720  # Độ phân giải hiển thị hợp lý
CONFIDENCE_BOOST = 1.2  # Hệ số tăng độ tin cậy khi hiển thị (120%)
FACE_ALIGNMENT = True  # Bật canh chỉnh khuôn mặt

# Thêm các biến cấu hình cho tính năng mới
# Cấu hình đa luồng và đa tiến trình
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Để lại 1 lõi cho hệ thống
THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
PROCESS_POOL = None  # Sẽ được khởi tạo khi cần

# Thêm cấu hình cho lưu trữ cache
DISK_CACHE_ENABLED = True
DISK_CACHE_FILE = os.path.join(TEMP_DIR, "face_cache.pkl")
DISK_CACHE_MAX_SIZE = 10000  # Số lượng khuôn mặt tối đa trong cache đĩa

# Thêm cấu hình cho Faiss/Annoy
EMBEDDING_DIM = 512  # ArcFace dimension
INDEX_REBUILD_THRESHOLD = 5  # Rebuild sau khi thêm 5 embedding mới

# Đảm bảo thư mục embeddings tồn tại
if not os.path.exists(EMBEDDINGS_DIR):
    os.makedirs(EMBEDDINGS_DIR)

# Đảm bảo thư mục known_faces tồn tại
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Khởi tạo mô hình nhận diện khuôn mặt
print("Đang khởi tạo mô hình nhận diện khuôn mặt...")
arcface_model = DeepFace.build_model("ArcFace")

# Thiết lập tham số
THRESHOLD = 0.5  # Tăng ngưỡng lên để giảm độ chặt chẽ trong so sánh khuôn mặt
DETECTION_INTERVAL = 1  # Xử lý mỗi frame để tăng độ mượt
FACE_CACHE_SIZE = 200  # Giảm cache size để tránh nhầm lẫn
RECOGNITION_BUFFER_SIZE = 8  # Giảm buffer size để phản ứng nhanh hơn 
MAX_FACES = 3  # Chỉ xử lý tối đa 3 khuôn mặt để tăng hiệu suất

# Hàm kiểm tra và tạo đường dẫn tạm duy nhất
def get_temp_path():
    timestamp = int(time.time() * 1000)
    thread_id = threading.get_ident()
    return os.path.join(TEMP_DIR, f"temp_{timestamp}_{thread_id}.jpg")

# Cập nhật hàm trích xuất embedding để đơn giản và hiệu quả hơn
# Cải tiến function extract_embedding
def extract_embedding(face_img):
    try:
        if face_img is None or face_img.size == 0:
            print("Face image is None or empty")
            return None
            
        # Đảm bảo ảnh là BGR
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
        
        # Kiểm tra kích thước ảnh đầu vào
        if face_img.shape[0] < 20 or face_img.shape[1] < 20:
            print(f"Face too small: {face_img.shape}")
            return None
            
        # Đơn giản hóa tiền xử lý để tránh lỗi
        # Chỉ resize và chuẩn hóa cơ bản
        face_img = cv2.resize(face_img, (112, 112))
        
        # Tạo file tạm với tên duy nhất
        temp_path = get_temp_path()
        cv2.imwrite(temp_path, face_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        try:
            # Sử dụng GPU nếu có
            with tf.device('/GPU:0') if gpus else tf.device('/CPU:0'):
                embedding_objs = DeepFace.represent(
                    img_path=temp_path,
                    model_name="ArcFace",
                    enforce_detection=False,
                    detector_backend="skip",
                    normalization="base",
                    align=False
                )
            
            # In ra thông tin và kiểm tra kết quả embedding
            if embedding_objs and len(embedding_objs) > 0:
                print(f"Embedding extracted successfully, shape: {len(embedding_objs[0]['embedding'])}")
            else:
                print("No embedding returned by DeepFace.represent")
            
            # Xóa file tạm
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            # Chuẩn hóa embedding (L2 normalization)
            if embedding_objs and len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]["embedding"])
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
            return None
            
        except Exception as e:
            print(f"Error in DeepFace.represent: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None
                
    except Exception as e:
        print(f"Error in extract_embedding: {e}")
        return None

# Hàm canh chỉnh khuôn mặt mới
def align_face(face_img):
    try:
        # Sử dụng dlib để phát hiện các điểm mốc trên khuôn mặt
        import dlib
        detector = dlib.get_frontal_face_detector()
        predictor_path = os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat')
        
        if os.path.exists(predictor_path):
            predictor = dlib.shape_predictor(predictor_path)
            
            # Chuyển đổi sang định dạng ảnh dlib
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            if len(faces) > 0:
                # Lấy điểm mốc cho khuôn mặt đầu tiên
                landmarks = predictor(gray, faces[0])
                
                # Lấy tọa độ mắt
                left_eye = [landmarks.part(36).x, landmarks.part(36).y]
                right_eye = [landmarks.part(45).x, landmarks.part(45).y]
                
                # Tính góc cần xoay
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Xoay ảnh để canh chỉnh theo mắt
                center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1)
                aligned_face = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
                
                return aligned_face
        
        # Nếu không tìm thấy file predictor hoặc quá trình canh chỉnh thất bại, trả về ảnh gốc
        return face_img
    except Exception:
        # Trả về ảnh gốc nếu có lỗi
        return face_img

# Hàm phát hiện khuôn mặt và tạo embedding - tối ưu cho tốc độ
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
        
    # Phát hiện khuôn mặt với RetinaFace
    try:
        # Giảm kích thước ảnh để tăng tốc xử lý
        scale_factor = min(1.0, 640 / max(image.shape[0], image.shape[1]))
        if scale_factor < 1.0:
            width = int(image.shape[1] * scale_factor)
            height = int(image.shape[0] * scale_factor)
            image = cv2.resize(image, (width, height))
        
        # Tiền xử lý để cải thiện khả năng phát hiện
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = cv2.convertScaleAbs(image, alpha=1.05, beta=5)
        
        # Phát hiện khuôn mặt
        with tf.device('/GPU:0') if gpus else tf.device('/CPU:0'):
            faces = RetinaFace.detect_faces(image, threshold=0.8)
            
        if not faces:  # Nếu không phát hiện khuôn mặt nào
            print(f"Không phát hiện khuôn mặt trong ảnh: {image_path}")
            return None
        
        # Lấy khuôn mặt có kích thước lớn nhất (thường là khuôn mặt chính)
        largest_area = 0
        largest_face_key = None
        
        for face_key in faces:
            face_data = faces[face_key]
            facial_area = face_data['facial_area']
            x1, y1, x2, y2 = facial_area
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_face_key = face_key
        
        if not largest_face_key:
            return None
        
        face_data = faces[largest_face_key]
        
        # Lấy tọa độ khuôn mặt
        x1, y1, x2, y2 = face_data['facial_area']
        
        # Kiểm tra tọa độ hợp lệ
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > image.shape[1]: x2 = image.shape[1]
        if y2 > image.shape[0]: y2 = image.shape[0]
        
        # Mở rộng vùng khuôn mặt một chút để lấy thêm ngữ cảnh
        h, w = image.shape[:2]
        expand = min(10, min(x1, y1, w-x2, h-y2))
        x1 = max(0, x1 - expand)
        y1 = max(0, y1 - expand)
        x2 = min(w, x2 + expand)
        y2 = min(h, y2 + expand)
        
        # Cắt khuôn mặt từ ảnh
        face_img = image[y1:y2, x1:x2]
        
        # Trích xuất embedding với ArcFace
        embedding = extract_embedding(face_img)
        return embedding
    
    except Exception as e:
        print(f"Lỗi khi phát hiện khuôn mặt: {e}")
        traceback.print_exc()
        return None

# Cache khuôn mặt đã nhận dạng với giới hạn kích thước
class FaceCache:
    def __init__(self, max_size=FACE_CACHE_SIZE):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.disk_cache = {}
        self.load_disk_cache()
    
    def get(self, face_hash):
        # Kiểm tra trong memory cache
        if face_hash in self.cache:
            # Di chuyển mục này lên đầu LRU cache
            value = self.cache.pop(face_hash)
            self.cache[face_hash] = value
            return value
        
        # Kiểm tra trong disk cache
        if DISK_CACHE_ENABLED and face_hash in self.disk_cache:
            value = self.disk_cache[face_hash]
            # Di chuyển từ disk cache lên memory cache
            self.cache[face_hash] = value
            # Nếu memory cache quá lớn, loại bỏ mục ít được sử dụng nhất
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
            return value
            
        return None
    
    def put(self, face_hash, value):
        # Thêm vào memory cache
        self.cache[face_hash] = value
        
        # Nếu memory cache quá lớn, loại bỏ mục ít được sử dụng nhất
        if len(self.cache) > self.max_size:
            oldest_key, oldest_value = self.cache.popitem(last=False)
            # Lưu mục bị loại bỏ vào disk cache
            if DISK_CACHE_ENABLED:
                self.disk_cache[oldest_key] = oldest_value
                
        # Lưu vào disk cache
        if DISK_CACHE_ENABLED:
            self.disk_cache[face_hash] = value
            # Lưu disk cache định kỳ (xác suất 1%)
            if np.random.random() < 0.01:
                self.save_disk_cache()
    
    def save_disk_cache(self):
        try:
            # Nếu disk cache quá lớn, chỉ giữ lại các mục mới nhất
            if len(self.disk_cache) > DISK_CACHE_MAX_SIZE:
                # Sắp xếp theo thời gian truy cập và chỉ giữ lại một phần
                pruned_cache = dict(list(self.disk_cache.items())[-DISK_CACHE_MAX_SIZE:])
                self.disk_cache = pruned_cache
                
            with open(DISK_CACHE_FILE, 'wb') as f:
                pickle.dump(self.disk_cache, f)
        except Exception as e:
            print(f"Lỗi khi lưu disk cache: {e}")
    
    def load_disk_cache(self):
        if DISK_CACHE_ENABLED and os.path.exists(DISK_CACHE_FILE):
            try:
                with open(DISK_CACHE_FILE, 'rb') as f:
                    self.disk_cache = pickle.load(f)
                print(f"Đã tải {len(self.disk_cache)} mục từ disk cache")
            except Exception as e:
                print(f"Lỗi khi tải disk cache: {e}")
                self.disk_cache = {}
        else:
            self.disk_cache = {}
    
    def clear_memory_cache(self):
        self.cache.clear()
    
    def size(self):
        return len(self.cache), len(self.disk_cache)

# Khởi tạo cache với kích thước được cập nhật
face_cache = FaceCache(max_size=FACE_CACHE_SIZE)

# Lớp vector index để tìm kiếm gần đúng hiệu quả
class VectorIndex:
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.names = []
        self.embeddings = []
        self.index = None
        self.annoy_index = None
        self.needs_rebuild = True
        self.built_size = 0
        
    def add_embedding(self, name, embedding):
        self.names.append(name)
        self.embeddings.append(embedding)
        # Đánh dấu để xây dựng lại index nếu cần
        if len(self.names) - self.built_size >= INDEX_REBUILD_THRESHOLD:
            self.needs_rebuild = True
        
    def build_index(self):
        if len(self.embeddings) == 0:
            print("Không có embedding để xây dựng index")
            return False
            
        if FAISS_AVAILABLE:
            try:
                # Chuẩn bị dữ liệu cho FAISS
                embeddings_array = np.array(self.embeddings).astype('float32')
                # Tạo chỉ mục
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product = cosine với vectors đã chuẩn hóa
                # Thêm vectors
                self.index.add(embeddings_array)
                self.built_size = len(self.embeddings)
                self.needs_rebuild = False
                print(f"Đã xây dựng FAISS index với {self.built_size} embeddings")
                return True
            except Exception as e:
                print(f"Lỗi khi xây dựng FAISS index: {e}")
                return False
        elif ANNOY_AVAILABLE:
            try:
                # Tạo chỉ mục Annoy
                self.annoy_index = AnnoyIndex(self.embedding_dim, 'angular')  # Angular = cosine distance
                # Thêm vectors
                for i, emb in enumerate(self.embeddings):
                    self.annoy_index.add_item(i, emb)
                # Xây dựng index với 10 trees (cân bằng tốc độ và độ chính xác)
                self.annoy_index.build(10)
                self.built_size = len(self.embeddings)
                self.needs_rebuild = False
                print(f"Đã xây dựng Annoy index với {self.built_size} embeddings")
                return True
            except Exception as e:
                print(f"Lỗi khi xây dựng Annoy index: {e}")
                return False
        else:
            # Không có thư viện tìm kiếm gần đúng, không cần xây dựng index
            self.built_size = len(self.embeddings)
            self.needs_rebuild = False
            return True
    
    def find_nearest(self, query_embedding, k=5):
        if len(self.embeddings) == 0:
            return [], []
            
        # Xây dựng lại index nếu cần
        if self.needs_rebuild:
            self.build_index()
            
        # Chuẩn hóa embedding trước khi tìm kiếm
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
        if FAISS_AVAILABLE and self.index is not None:
            try:
                # Tìm kiếm k vectors gần nhất
                query_array = np.array([query_embedding]).astype('float32')
                distances, indices = self.index.search(query_array, min(k, len(self.embeddings)))
                
                # Chuyển đổi kết quả
                result_names = [self.names[idx] for idx in indices[0]]
                result_distances = [1 - dist for dist in distances[0]]  # Chuyển từ similarity sang distance
                
                return result_names, result_distances
            except Exception as e:
                print(f"Lỗi khi tìm kiếm FAISS: {e}")
                # Fallback sang tìm kiếm tuần tự
        
        elif ANNOY_AVAILABLE and self.annoy_index is not None:
            try:
                # Tìm kiếm k vectors gần nhất
                indices, distances = self.annoy_index.get_nns_by_vector(
                    query_embedding, min(k, len(self.embeddings)), 
                    include_distances=True
                )
                
                # Chuyển đổi kết quả
                result_names = [self.names[idx] for idx in indices]
                result_distances = distances  # Annoy trả về khoảng cách trực tiếp
                
                return result_names, result_distances
            except Exception as e:
                print(f"Lỗi khi tìm kiếm Annoy: {e}")
                # Fallback sang tìm kiếm tuần tự
        
        # Tìm kiếm tuần tự
        distances = []
        for emb in self.embeddings:
            # Chuẩn hóa embedding từ cơ sở dữ liệu
            emb_norm = emb / np.linalg.norm(emb)
            # Tính khoảng cách cosine
            dist = cosine(query_embedding, emb_norm)
            distances.append(dist)
            
        # Tìm k vectors có khoảng cách nhỏ nhất
        indices = np.argsort(distances)[:k]
        result_names = [self.names[idx] for idx in indices]
        result_distances = [distances[idx] for idx in indices]
        
        return result_names, result_distances
    
    def save(self, path):
        data = {
            'names': self.names,
            'embeddings': self.embeddings,
            'embedding_dim': self.embedding_dim
        }
        try:
            joblib.dump(data, path)
            print(f"Đã lưu vector index với {len(self.names)} embeddings")
            return True
        except Exception as e:
            print(f"Lỗi khi lưu vector index: {e}")
            return False
    
    def load(self, path):
        if os.path.exists(path):
            try:
                data = joblib.load(path)
                self.names = data['names']
                self.embeddings = data['embeddings']
                self.embedding_dim = data.get('embedding_dim', EMBEDDING_DIM)
                self.needs_rebuild = True
                print(f"Đã tải vector index với {len(self.names)} embeddings")
                return True
            except Exception as e:
                print(f"Lỗi khi tải vector index: {e}")
                return False
        return False

# Khởi tạo vector index toàn cục
vector_index = VectorIndex()
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "face_index.joblib")

# Cải tiến hàm so sánh khuôn mặt sử dụng tìm kiếm gần đúng
def compare_faces(embedding, known_faces):
    if not known_faces:
        print("No known faces to compare with")
        return "Unknown", None
        
    # Kiểm tra embedding đầu vào
    if embedding is None or len(embedding) == 0:
        print("Input embedding is None or empty")
        return "Unknown", None
        
    # Debug: in số lượng khuôn mặt đã biết để kiểm tra
    print(f"Comparing with {len(known_faces)} known faces")
    
    # Tìm kiếm gần đúng với vector index
    if vector_index and len(vector_index.names) > 0:
        try:
            # Tìm kiếm 3 vectors gần nhất
            result_names, result_distances = vector_index.find_nearest(embedding, k=3)
            
            if result_names and result_distances:
                closest_name = result_names[0]
                min_distance = result_distances[0]
                
                # In ra thông tin khoảng cách và ngưỡng
                print(f"Best match: {closest_name} with distance {min_distance:.4f} (threshold: {THRESHOLD})")
                
                # Kiểm tra xem khoảng cách có nhỏ hơn ngưỡng không
                if min_distance < THRESHOLD:
                    return closest_name, min_distance
            
            return "Unknown", None
        except Exception as e:
            print(f"Error in vector search: {e}")
            # Fallback sang phương pháp so sánh truyền thống
    
    # Phương pháp so sánh truyền thống nếu vector index không khả dụng
    try:
        # Chuẩn hóa embedding đầu vào một lần duy nhất
        embedding_norm = embedding / np.linalg.norm(embedding)
        
        # Sử dụng đa luồng để tính toán khoảng cách
        def calculate_distance(args):
            name, known_embedding = args
            if known_embedding is None or len(known_embedding) == 0:
                return name, float('inf')
            # Chuẩn hóa embedding đã biết
            known_norm = known_embedding / np.linalg.norm(known_embedding)
            # Tính khoảng cách cosine
            distance = cosine(embedding_norm, known_norm)
            return name, distance
        
        # Tính toán khoảng cách song song
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            distance_results = list(executor.map(calculate_distance, known_faces.items()))
        
        # Lọc kết quả không hợp lệ
        valid_results = [(name, dist) for name, dist in distance_results if dist != float('inf')]
        
        # Nếu không có kết quả hợp lệ
        if not valid_results:
            return "Unknown", None
        
        # Tìm khoảng cách nhỏ nhất
        closest = min(valid_results, key=lambda x: x[1])
        closest_name, min_distance = closest
        
        # In ra thông tin khoảng cách và ngưỡng
        print(f"Best match: {closest_name} with distance {min_distance:.4f} (threshold: {THRESHOLD})")
        
        # Kiểm tra xem khoảng cách có nhỏ hơn ngưỡng không
        if min_distance < THRESHOLD:
            return closest_name, min_distance
        
    except Exception as e:
        print(f"Error during face comparison: {e}")
    
    return "Unknown", None

# Cập nhật hàm xử lý thư mục ảnh người quen để tạo vector index
def process_known_faces():
    print("\n===== Xử lý khuôn mặt đã biết =====")
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR)

    # Tạo mới vector index
    global vector_index
    vector_index = VectorIndex()

    # Xử lý hình ảnh song song
    def process_person(person_name):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            return None, 0, 0
            
        print(f"Đang xử lý dữ liệu của: {person_name}")
        embeddings = []
        image_count = 0
        success_count = 0
        
        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(person_dir, image_name)
            image_count += 1
            print(f"  Đang xử lý ảnh: {image_name}", end="... ")
            
            embedding = process_image(image_path)
            if embedding is not None:
                embeddings.append(embedding)
                success_count += 1
                print("Thành công")
            else:
                print("Thất bại")
                
        if embeddings:
            # Tính embedding trung bình
            avg_embedding = np.mean(embeddings, axis=0)
            # Chuẩn hóa embedding trước khi lưu
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            return person_name, avg_embedding, image_count, success_count
        
        return None, 0, 0

    # Thu thập danh sách người dùng
    person_names = [d for d in os.listdir(KNOWN_FACES_DIR) 
                   if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]
    
    # Xử lý song song với ProcessPoolExecutor
    person_count = 0
    results = []
    
    # Khởi tạo process pool nếu chưa có
    global PROCESS_POOL
    if PROCESS_POOL is None:
        PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)
    
    # Xử lý từng người dùng (không song song hóa vì quá trình này đã khá nhanh)
    for person_name in person_names:
        result = process_person(person_name)
        if result[0]:  # Nếu có kết quả hợp lệ
            name, embedding, image_count, success_count = result
            
            # Lưu embedding vào file
            embedding_path = os.path.join(EMBEDDINGS_DIR, f"{name}.npy")
            np.save(embedding_path, embedding)
            
            # Thêm vào vector index
            vector_index.add_embedding(name, embedding)
            
            # Kiểm tra file đã lưu
            if os.path.exists(embedding_path):
                filesize = os.path.getsize(embedding_path)
                print(f"✓ Đã lưu embedding cho {name} ({success_count}/{image_count} ảnh thành công) - File size: {filesize} bytes")
                person_count += 1
            else:
                print(f"✗ Lỗi: Không thể lưu file embedding cho {name}")
        else:
            print(f"✗ Không thể tạo embedding cho {person_names[i]} - không có khuôn mặt được phát hiện")
    
    # Xây dựng vector index
    if person_count > 0:
        vector_index.build_index()
        # Lưu vector index
        vector_index.save(INDEX_PATH)
    
    print(f"\nHoàn tất xử lý cho {person_count} người dùng.")
    input("Nhấn Enter để tiếp tục...")

# Hàm load vector index 
def load_face_database():
    # Tạo danh sách khuôn mặt đã biết
    known_faces = {}
    for file_name in os.listdir(EMBEDDINGS_DIR):
        if file_name.endswith(".npy"):
            name = os.path.splitext(file_name)[0]
            embedding_path = os.path.join(EMBEDDINGS_DIR, file_name)
            known_faces[name] = np.load(embedding_path)
    
    # Tạo vector index từ danh sách khuôn mặt đã biết
    global vector_index
    if vector_index.load(INDEX_PATH):
        print("Đã tải vector index từ file")
    else:
        print("Đang xây dựng vector index mới...")
        vector_index = VectorIndex()
        for name, embedding in known_faces.items():
            vector_index.add_embedding(name, embedding)
        vector_index.build_index()
        vector_index.save(INDEX_PATH)
    
    return known_faces

# Bộ đệm kết quả nhận dạng để ổn định nhận diện qua nhiều frame
class RecognitionBuffer:
    def __init__(self, size=RECOGNITION_BUFFER_SIZE):
        self.size = size
        self.buffer = []
        self.last_stable_result = ("Unknown", None)
        self.stable_count = 0
        
    def update(self, name, distance):
        self.buffer.append((name, distance))
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
    
    def get_result(self):
        if not self.buffer:
            return "Unknown", None
        
        # Đếm số lần xuất hiện của mỗi tên
        name_counts = {}
        for name, distance in self.buffer:
            if name == "Unknown":
                continue
            if name not in name_counts:
                name_counts[name] = []
            name_counts[name].append(distance)
        
        # Nếu không có tên nào được xác định
        if not name_counts:
            return "Unknown", None
        
        # Tìm tên xuất hiện nhiều nhất
        top_name = max(name_counts.items(), key=lambda x: len(x[1]))
        name, distances = top_name
        
        # Chỉ chấp nhận kết quả nếu xuất hiện đủ số lần
        threshold = max(2, self.size // 3)  # Tối thiểu 2 lần hoặc 1/3 buffer size
        
        if len(distances) >= threshold:
            # Tính khoảng cách trung bình
            avg_distance = sum(distances) / len(distances)
            
            # Tăng độ ổn định: chỉ thay đổi kết quả khi có đủ bằng chứng
            if name == self.last_stable_result[0]:
                self.stable_count += 1
            else:
                self.stable_count = 1
                self.last_stable_result = (name, avg_distance)
            
            # Nếu đã ổn định qua nhiều frame
            if self.stable_count > 2:
                return self.last_stable_result
            
            return name, avg_distance
            
        # Nếu không có kết quả nào đủ ổn định, giữ nguyên kết quả cũ
        if self.last_stable_result[0] != "Unknown":
            self.stable_count -= 1
            # Chỉ giữ kết quả cũ trong một khoảng thời gian nhất định
            if self.stable_count > 0:
                return self.last_stable_result
        
        return "Unknown", None

# Cập nhật thread xử lý nhận dạng khuôn mặt để xử lý song song
def recognition_thread(frame, faces, known_faces, recognition_buffers):
    global latest_recognition, processing
    
    try:
        if not faces:
            print("No faces detected in frame")
            processing = False
            return
            
        print(f"Processing {len(faces)} detected faces")
        
        # Chuẩn bị danh sách khuôn mặt và tọa độ
        face_data_list = []
        face_imgs = []
        
        # Sắp xếp theo kích thước giảm dần
        sorted_faces = []
        for face_key in faces:
            face_data = faces[face_key]
            facial_area = face_data['facial_area']
            x1, y1, x2, y2 = facial_area
            area = (x2 - x1) * (y2 - y1)
            sorted_faces.append((face_key, area, facial_area))
        
        sorted_faces.sort(key=lambda x: x[1], reverse=True)
        
        # Giới hạn số lượng khuôn mặt xử lý
        for face_key, area, facial_area in sorted_faces[:MAX_FACES]:
            x1, y1, x2, y2 = facial_area
            
            # Kiểm tra kích thước khuôn mặt
            if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and x2 < frame.shape[1] and y2 < frame.shape[0]:
                # Cắt khuôn mặt
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.shape[0] >= 30 and face_img.shape[1] >= 30:
                    # Lưu thông tin khuôn mặt
                    face_data_list.append((face_key, facial_area))
                    face_imgs.append(face_img)
        
        # Nếu không có khuôn mặt hợp lệ
        if not face_imgs:
            processing = False
            return
            
        # Khởi tạo kết quả
        results = []
        
        # Xử lý từng khuôn mặt song song
        def process_face(idx):
            face_key, facial_area = face_data_list[idx]
            face_img = face_imgs[idx]
            
            # Tính hash nhanh
            resized_face = cv2.resize(face_img, (16, 16))
            face_hash = hash(resized_face.tobytes())
            
            # Kiểm tra cache
            cache_result = face_cache.get(face_hash)
            if cache_result:
                print("Using cached recognition result")
                name, distance = cache_result
            else:
                # Trích xuất embedding
                embedding = extract_embedding(face_img)
                
                if embedding is not None:
                    name, distance = compare_faces(embedding, known_faces)
                    if name != "Unknown":
                        print(f"Recognized: {name} with distance {distance}")
                        face_cache.put(face_hash, (name, distance))
                else:
                    name, distance = "Unknown", None
            
            # Cập nhật buffer
            if face_key not in recognition_buffers:
                recognition_buffers[face_key] = RecognitionBuffer(size=RECOGNITION_BUFFER_SIZE)
            recognition_buffers[face_key].update(name, distance)
            
            # Lấy kết quả từ buffer
            stable_name, stable_distance = recognition_buffers[face_key].get_result()
            
            return facial_area, stable_name, stable_distance
        
        # Xử lý tất cả khuôn mặt song song
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_FACES) as executor:
            for result in executor.map(process_face, range(len(face_data_list))):
                results.append(result)
        
        # Cập nhật kết quả
        latest_recognition = results
    
    except Exception as e:
        print(f"Error in recognition thread: {e}")
    
    finally:
        processing = False

# Cập nhật hàm nhận diện khuôn mặt để load và sử dụng vector index
def recognize_faces():
    print("\n===== Bắt đầu nhận dạng thời gian thực =====")
    
    # Load embedding của người quen và tạo vector index
    known_faces = load_face_database()
    
    # Thông báo nếu không có dữ liệu người quen
    person_count = len(known_faces)
    if person_count == 0:
        print("Không có dữ liệu người quen nào. Vui lòng xử lý khuôn mặt đã biết trước!")
        input("Nhấn Enter để tiếp tục...")
        return
    else:
        print(f"Đã tải {person_count} người quen vào bộ nhớ")
    
    # Khởi động camera
    print("Đang khởi động camera...")
    cap = cv2.VideoCapture(0)
    
    # Thiết lập camera để cân bằng độ phân giải và FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Độ phân giải trung bình
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm độ trễ
    
    if not cap.isOpened():
        print("Không thể mở camera. Vui lòng kiểm tra lại kết nối.")
        input("Nhấn Enter để tiếp tục...")
        return
    
    print("Camera đã sẵn sàng. Nhấn 'q' để thoát, 's' để chụp ảnh màn hình")
    
    # Khởi tạo các biến theo dõi hiệu suất
    fps_start_time = time.time()
    frame_count = 0
    process_this_frame = 0
    fps = 0.0
    avg_processing_time = 0.0
    processing_times = []
    
    # Reset biến global
    global latest_recognition, processing
    latest_recognition = []
    processing = False
    recognition_buffers = {}
    
    # Thư mục lưu ảnh chụp màn hình
    screenshots_dir = os.path.join(os.getcwd(), "screenshots")
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)
    
    # Vòng lặp chính xử lý video
    while True:
        # Bắt đầu đo thời gian xử lý frame
        frame_start_time = time.time()
        
        # Đọc frame từ camera
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera. Thoát...")
            break

        # Tính FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 0.5:  # Cập nhật FPS 2 lần mỗi giây
            fps = frame_count / elapsed_time
            frame_count = 0
            fps_start_time = time.time()

        # Chỉ xử lý mỗi DETECTION_INTERVAL frame để cải thiện hiệu suất
        process_this_frame = (process_this_frame + 1) % DETECTION_INTERVAL
        
        # Tạo frame hiển thị với tỷ lệ khung hình đúng
        h, w = frame.shape[:2]
        display_height = int(h * DISPLAY_WIDTH / w)
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, display_height))
        
        # Xử lý phát hiện và nhận dạng khuôn mặt
        if process_this_frame == 0 and not processing:
            processing = True
            process_start = time.time()
            
            try:
                # Phát hiện khuôn mặt trên frame nhỏ hơn để tăng tốc
                detection_frame = cv2.resize(frame, (DETECTION_SIZE, DETECTION_SIZE))
                
                # Tăng tương phản để cải thiện khả năng phát hiện
                detection_frame = cv2.convertScaleAbs(detection_frame, alpha=1.3, beta=10)
                
                # Thực hiện phát hiện khuôn mặt
                with tf.device('/GPU:0') if gpus else tf.device('/CPU:0'):
                    faces = RetinaFace.detect_faces(detection_frame, threshold=0.9)
                
                # Xử lý các khuôn mặt được phát hiện
                if faces:
                    # Tỉ lệ để chuyển từ khung phát hiện sang khung gốc
                    scale_x = w / DETECTION_SIZE
                    scale_y = h / DETECTION_SIZE
                    
                    # Sắp xếp khuôn mặt theo diện tích giảm dần
                    sorted_faces = []
                    for face_key in faces:
                        face_data = faces[face_key]
                        facial_area = face_data['facial_area']
                        x1, y1, x2, y2 = facial_area
                        
                        # Scale về kích thước gốc
                        x1_orig = int(x1 * scale_x)
                        y1_orig = int(y1 * scale_y)
                        x2_orig = int(x2 * scale_x)
                        y2_orig = int(y2 * scale_y)
                        
                        # Đảm bảo tọa độ nằm trong frame
                        x1_orig = max(0, min(x1_orig, w-1))
                        y1_orig = max(0, min(y1_orig, h-1))
                        x2_orig = max(x1_orig+1, min(x2_orig, w))
                        y2_orig = max(y1_orig+1, min(y2_orig, h))
                        
                        area = (x2_orig - x1_orig) * (y2_orig - y1_orig)
                        sorted_faces.append((face_key, area, (x1_orig, y1_orig, x2_orig, y2_orig)))
                    
                    # Sắp xếp theo diện tích và giới hạn số khuôn mặt xử lý
                    sorted_faces.sort(key=lambda x: x[1], reverse=True)
                    face_results = []
                    
                    # Chỉ xử lý MAX_FACES khuôn mặt lớn nhất
                    for face_idx, (face_key, _, facial_area) in enumerate(sorted_faces[:MAX_FACES]):
                        x1, y1, x2, y2 = facial_area
                        
                        # Bỏ qua khuôn mặt quá nhỏ
                        if (x2 - x1) < 30 or (y2 - y1) < 30:
                            continue
                            
                        # Cắt khuôn mặt từ frame gốc
                        face_img = frame[y1:y2, x1:x2]
                        
                        # Tính hash nhanh cho việc cache
                        resized_face = cv2.resize(face_img, (16, 16))
                        face_hash = hash(resized_face.tobytes())
                        
                        # Nhận dạng khuôn mặt
                        name = "Unknown"
                        distance = None
                        
                        # Kiểm tra cache trước
                        cache_result = face_cache.get(face_hash)
                        if cache_result:
                            name, distance = cache_result
                        else:
                            # Trích xuất embedding chỉ khi cần
                            embedding = extract_embedding(face_img)
                            if embedding is not None:
                                name, distance = compare_faces(embedding, known_faces)
                                # Chỉ cache kết quả người quen để tiết kiệm bộ nhớ
                                if name != "Unknown":
                                    face_cache.put(face_hash, (name, distance))
                        
                        # Sử dụng buffer riêng cho mỗi vị trí khuôn mặt
                        buffer_key = f"face_{face_idx}"
                        if buffer_key not in recognition_buffers:
                            recognition_buffers[buffer_key] = RecognitionBuffer(size=RECOGNITION_BUFFER_SIZE)
                            
                        recognition_buffers[buffer_key].update(name, distance)
                        stable_name, stable_distance = recognition_buffers[buffer_key].get_result()
                        
                        # Chuyển đổi tọa độ sang frame hiển thị
                        display_x1 = int(x1 * DISPLAY_WIDTH / w)
                        display_y1 = int(y1 * display_height / h)
                        display_x2 = int(x2 * DISPLAY_WIDTH / w)
                        display_y2 = int(y2 * display_height / h)
                        
                        face_results.append(((display_x1, display_y1, display_x2, display_y2), 
                                           stable_name, stable_distance))
                    
                    # Cập nhật kết quả nhận diện
                    latest_recognition = face_results
                
                # Ghi lại thời gian xử lý để phân tích hiệu suất
                process_time = time.time() - process_start
                processing_times.append(process_time)
                if len(processing_times) > 30:  # Giữ thống kê 30 frame gần nhất
                    processing_times.pop(0)
                avg_processing_time = sum(processing_times) / len(processing_times)
            
            except Exception as e:
                print(f"Lỗi phát hiện khuôn mặt: {str(e)[:50]}...")
            
            processing = False
        
        # Hiển thị kết quả nhận dạng
        for facial_area, name, distance in latest_recognition:
            try:
                x1, y1, x2, y2 = facial_area
                
                # Xác định màu và nhãn dựa trên kết quả nhận dạng
                if name == "Unknown":
                    color = (0, 0, 255)  # Đỏ cho người lạ
                    label = "Unknown"
                else:
                    # Tính độ tin cậy với sự tăng cường
                    confidence_pct = min(100, max(70, int((1 - distance/THRESHOLD) * CONFIDENCE_BOOST * 100)))
                    
                    # Màu dựa trên độ tin cậy (từ xanh lá đến vàng)
                    if confidence_pct > 85:  # Tin cậy cao
                        color = (0, 255, 0)  # Xanh lá
                    elif confidence_pct > 75:  # Tin cậy trung bình
                        color = (0, 255, 255)  # Vàng
                    else:  # Tin cậy thấp
                        color = (0, 165, 255)  # Cam
                    
                    label = f"{name} ({confidence_pct}%)"
                
                # Vẽ khung xung quanh khuôn mặt
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Vẽ nền cho nhãn
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
                cv2.rectangle(display_frame, 
                              (x1, y1 - 30), 
                              (x1 + label_size[0] + 10, y1), 
                              color, cv2.FILLED)
                
                # Hiển thị nhãn
                cv2.putText(display_frame, label, (x1 + 5, y1 - 8), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception:
                pass
        
        # Hiển thị thống kê về hiệu năng
        try:
            # Tạo thanh trạng thái ở trên cùng
            status_height = 50
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], status_height), (0, 0, 0), cv2.FILLED)
            
            # FPS
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(display_frame, fps_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Thời gian xử lý
            time_text = f"Process: {avg_processing_time*1000:.1f}ms"
            cv2.putText(display_frame, time_text, (180, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Trạng thái GPU và model
            if gpus:
                gpu_text = "GPU: Active"
                color = (0, 255, 0)
            else:
                gpu_text = "GPU: Inactive"
                color = (0, 100, 255)
                
            cv2.putText(display_frame, gpu_text, (350, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            model_text = "Model: " + ("Facenet" if USE_LIGHTWEIGHT_MODEL else "ArcFace")
            cv2.putText(display_frame, model_text, (500, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Hiển thị số người đã biết
            cv2.putText(display_frame, f"DB: {person_count} người", 
                       (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        except Exception:
            pass
        
        # Hiển thị frame kết quả
        try:
            cv2.imshow("Face Recognition", display_frame)
        except Exception:
            break
        
        # Xử lý phím nhấn
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Thoát
            break
        elif key == ord('s'):  # Chụp màn hình
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            screenshot_path = os.path.join(screenshots_dir, f"capture_{timestamp}.jpg")
            cv2.imwrite(screenshot_path, display_frame)
            print(f"Đã lưu ảnh chụp màn hình tại: {screenshot_path}")

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

# Hàm thêm người quen mới bằng camera
def add_new_person():
    print("\n===== Thêm người quen mới bằng camera =====")
    
    # Nhập tên người mới
    while True:
        name = input("Nhập tên người mới (không dấu, dùng dấu '_' thay cho khoảng trắng): ").strip()
        if name and all(c.isalnum() or c == '_' for c in name):
            break
        print("Tên không hợp lệ. Vui lòng chỉ sử dụng chữ cái, số và dấu gạch dưới (_).")
    
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if os.path.exists(person_dir):
        choice = input(f"Thư mục cho {name} đã tồn tại. Bạn muốn ghi đè không? (y/n): ").lower()
        if choice != 'y':
            print("Hủy thêm người mới.")
            input("Nhấn Enter để tiếp tục...")
            return
    else:
        os.makedirs(person_dir)

    # Cải thiện cài đặt camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer size để giảm độ trễ
    
    if not cap.isOpened():
        print("Không thể mở camera. Vui lòng kiểm tra lại kết nối.")
        input("Nhấn Enter để tiếp tục...")
        return

    count = 0
    max_count = 10  # Số ảnh tối đa
    
    print(f"Thêm mới người dùng: {name}")
    print("Nhấn phím 'c' để chụp ảnh, 'q' để thoát.")
    print(f"Cần chụp {max_count} ảnh với các góc độ khác nhau để tăng độ chính xác.")

    while count < max_count:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera. Thoát...")
            break

        # Phát hiện khuôn mặt
        face_detected = False
        try:
            # Tiền xử lý ảnh để cải thiện khả năng phát hiện
            processed_frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)
            small_frame = cv2.resize(processed_frame, (0, 0), fx=0.5, fy=0.5)
            
            # Phát hiện khuôn mặt với GPU nếu có
            with tf.device('/GPU:0') if gpus else tf.device('/CPU:0'):
                faces = RetinaFace.detect_faces(small_frame)
            
            largest_area = 0
            largest_face = None
            
            if faces:
                for face_key in faces:
                    face_data = faces[face_key]
                    facial_area = face_data['facial_area']
                    x1, y1, x2, y2 = facial_area
                    area = (x2 - x1) * (y2 - y1)
                    
                    if area > largest_area:
                        largest_area = area
                        largest_face = (int(x1*2), int(y1*2), int(x2*2), int(y2*2))
                
                if largest_face:
                    x1, y1, x2, y2 = largest_face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    face_detected = True
        except Exception as e:
            print(f"Lỗi khi phát hiện khuôn mặt: {e}")
        
        # Hiển thị hướng dẫn và trạng thái
        status_text = f"Ảnh đã chụp: {count}/{max_count}" 
        guide_text = "Di chuyển khuôn mặt để có các góc nhìn khác nhau"
        detect_text = "ĐÃ PHÁT HIỆN KHUÔN MẶT" if face_detected else "KHÔNG PHÁT HIỆN KHUÔN MẶT"
        detect_color = (0, 255, 0) if face_detected else (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, guide_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, detect_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, detect_color, 2)
        
        cv2.imshow("Thêm người mới", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and face_detected:  # Chụp ảnh khi phát hiện khuôn mặt
            image_path = os.path.join(person_dir, f"{name}_{count:03d}.jpg")
            
            # Nâng cao chất lượng ảnh lưu trữ
            enhanced_frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=10)
            cv2.imwrite(image_path, enhanced_frame)
            
            print(f"Đã lưu ảnh: {image_path}")
            count += 1
        elif key == ord('q'):  # Thoát
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if count > 0:
        print(f"Đã chụp {count} ảnh cho {name}.")
        print("Bạn có muốn xử lý ngay bây giờ không?")
        choice = input("Xử lý ngay? (y/n): ").lower()
        if choice == 'y':
            process_known_faces()
        else:
            print("Bạn có thể xử lý sau bằng tùy chọn 1 trong menu chính.")
    else:
        print("Không có ảnh nào được chụp.")
    
    input("Nhấn Enter để tiếp tục...")

# Hàm xử lý thư mục ảnh người quen
def process_known_faces():
    print("\n===== Xử lý khuôn mặt đã biết =====")
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR)

    # Tạo mới vector index
    global vector_index
    vector_index = VectorIndex()

    # Xử lý hình ảnh song song
    def process_person(person_name):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            return None, 0, 0
            
        print(f"Đang xử lý dữ liệu của: {person_name}")
        embeddings = []
        image_count = 0
        success_count = 0
        
        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(person_dir, image_name)
            image_count += 1
            print(f"  Đang xử lý ảnh: {image_name}", end="... ")
            
            embedding = process_image(image_path)
            if embedding is not None:
                embeddings.append(embedding)
                success_count += 1
                print("Thành công")
            else:
                print("Thất bại")
                
        if embeddings:
            # Tính embedding trung bình
            avg_embedding = np.mean(embeddings, axis=0)
            # Chuẩn hóa embedding trước khi lưu
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            return person_name, avg_embedding, image_count, success_count
        
        return None, 0, 0

    # Thu thập danh sách người dùng
    person_names = [d for d in os.listdir(KNOWN_FACES_DIR) 
                   if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]
    
    # Xử lý song song với ProcessPoolExecutor
    person_count = 0
    results = []
    
    # Khởi tạo process pool nếu chưa có
    global PROCESS_POOL
    if PROCESS_POOL is None:
        PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)
    
    # Xử lý từng người dùng (không song song hóa vì quá trình này đã khá nhanh)
    for person_name in person_names:
        result = process_person(person_name)
        if result[0]:  # Nếu có kết quả hợp lệ
            name, embedding, image_count, success_count = result
            
            # Lưu embedding vào file
            embedding_path = os.path.join(EMBEDDINGS_DIR, f"{name}.npy")
            np.save(embedding_path, embedding)
            
            # Thêm vào vector index
            vector_index.add_embedding(name, embedding)
            
            # Kiểm tra file đã lưu
            if os.path.exists(embedding_path):
                filesize = os.path.getsize(embedding_path)
                print(f"✓ Đã lưu embedding cho {name} ({success_count}/{image_count} ảnh thành công) - File size: {filesize} bytes")
                person_count += 1
            else:
                print(f"✗ Lỗi: Không thể lưu file embedding cho {name}")
        else:
            print(f"✗ Không thể tạo embedding cho {person_names[i]} - không có khuôn mặt được phát hiện")
    
    # Xây dựng vector index
    if person_count > 0:
        vector_index.build_index()
        # Lưu vector index
        vector_index.save(INDEX_PATH)
    
    print(f"\nHoàn tất xử lý cho {person_count} người dùng.")
    input("Nhấn Enter để tiếp tục...")

# Hàm load vector index 
def load_face_database():
    # Tạo danh sách khuôn mặt đã biết
    known_faces = {}
    for file_name in os.listdir(EMBEDDINGS_DIR):
        if file_name.endswith(".npy"):
            name = os.path.splitext(file_name)[0]
            embedding_path = os.path.join(EMBEDDINGS_DIR, file_name)
            known_faces[name] = np.load(embedding_path)
    
    # Tạo vector index từ danh sách khuôn mặt đã biết
    global vector_index
    if vector_index.load(INDEX_PATH):
        print("Đã tải vector index từ file")
    else:
        print("Đang xây dựng vector index mới...")
        vector_index = VectorIndex()
        for name, embedding in known_faces.items():
            vector_index.add_embedding(name, embedding)
        vector_index.build_index()
        vector_index.save(INDEX_PATH)
    
    return known_faces

# Cập nhật hàm trích xuất embedding để chạy song song hiệu quả hơn
def extract_embedding_batch(face_imgs):
    """Trích xuất embedding cho một batch khuôn mặt"""
    results = []
    temp_paths = []
    
    try:
        # Xử lý hàng loạt hình ảnh
        for face_img in face_imgs:
            if face_img is None or face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                results.append(None)
                continue
                
            # Đảm bảo ảnh là BGR
            if len(face_img.shape) == 2:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
            
            # Resize khuôn mặt
            face_img = cv2.resize(face_img, (112, 112))
            
            # Tạo file tạm với tên duy nhất
            temp_path = get_temp_path()
            temp_paths.append(temp_path)
            cv2.imwrite(temp_path, face_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        # Xử lý song song với GPU
        with tf.device('/GPU:0') if gpus else tf.device('/CPU:0'):
            embeddings = []
            for temp_path in temp_paths:
                try:
                    embedding_objs = DeepFace.represent(
                        img_path=temp_path,
                        model_name="ArcFace",
                        enforce_detection=False,
                        detector_backend="skip",
                        normalization="base",
                        align=False
                    )
                    
                    if embedding_objs and len(embedding_objs) > 0:
                        # Chuẩn hóa embedding
                        embedding = np.array(embedding_objs[0]["embedding"])
                        embedding = embedding / np.linalg.norm(embedding)
                        embeddings.append(embedding)
                    else:
                        embeddings.append(None)
                except:
                    embeddings.append(None)
            
            results = embeddings
    except Exception as e:
        print(f"Error in batch embedding extraction: {e}")
    
    # Dọn dẹp các file tạm
    for temp_path in temp_paths:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
    
    return results

# Cập nhật thread xử lý nhận dạng khuôn mặt để xử lý song song
def recognition_thread(frame, faces, known_faces, recognition_buffers):
    global latest_recognition, processing
    
    try:
        if not faces:
            print("No faces detected in frame")
            processing = False
            return
            
        print(f"Processing {len(faces)} detected faces")
        
        # Chuẩn bị danh sách khuôn mặt và tọa độ
        face_data_list = []
        face_imgs = []
        
        # Sắp xếp theo kích thước giảm dần
        sorted_faces = []
        for face_key in faces:
            face_data = faces[face_key]
            facial_area = face_data['facial_area']
            x1, y1, x2, y2 = facial_area
            area = (x2 - x1) * (y2 - y1)
            sorted_faces.append((face_key, area, facial_area))
        
        sorted_faces.sort(key=lambda x: x[1], reverse=True)
        
        # Giới hạn số lượng khuôn mặt xử lý
        for face_key, area, facial_area in sorted_faces[:MAX_FACES]:
            x1, y1, x2, y2 = facial_area
            
            # Kiểm tra kích thước khuôn mặt
            if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and x2 < frame.shape[1] and y2 < frame.shape[0]:
                # Cắt khuôn mặt
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.shape[0] >= 30 and face_img.shape[1] >= 30:
                    # Lưu thông tin khuôn mặt
                    face_data_list.append((face_key, facial_area))
                    face_imgs.append(face_img)
        
        # Nếu không có khuôn mặt hợp lệ
        if not face_imgs:
            processing = False
            return
            
        # Khởi tạo kết quả
        results = []
        
        # Xử lý từng khuôn mặt song song
        def process_face(idx):
            face_key, facial_area = face_data_list[idx]
            face_img = face_imgs[idx]
            
            # Tính hash nhanh
            resized_face = cv2.resize(face_img, (16, 16))
            face_hash = hash(resized_face.tobytes())
            
            # Kiểm tra cache
            cache_result = face_cache.get(face_hash)
            if cache_result:
                print("Using cached recognition result")
                name, distance = cache_result
            else:
                # Trích xuất embedding
                embedding = extract_embedding(face_img)
                
                if embedding is not None:
                    name, distance = compare_faces(embedding, known_faces)
                    if name != "Unknown":
                        print(f"Recognized: {name} with distance {distance}")
                        face_cache.put(face_hash, (name, distance))
                else:
                    name, distance = "Unknown", None
            
            # Cập nhật buffer
            if face_key not in recognition_buffers:
                recognition_buffers[face_key] = RecognitionBuffer(size=RECOGNITION_BUFFER_SIZE)
            recognition_buffers[face_key].update(name, distance)
            
            # Lấy kết quả từ buffer
            stable_name, stable_distance = recognition_buffers[face_key].get_result()
            
            return facial_area, stable_name, stable_distance
        
        # Xử lý tất cả khuôn mặt song song
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_FACES) as executor:
            for result in executor.map(process_face, range(len(face_data_list))):
                results.append(result)
        
        # Cập nhật kết quả
        latest_recognition = results
    
    except Exception as e:
        print(f"Error in recognition thread: {e}")
    
    finally:
        processing = False

# Cập nhật hàm nhận diện khuôn mặt để load và sử dụng vector index
def recognize_faces():
    print("\n===== Bắt đầu nhận dạng thời gian thực =====")
    
    # Load embedding của người quen và tạo vector index
    known_faces = load_face_database()
    
    # Thông báo nếu không có dữ liệu người quen
    person_count = len(known_faces)
    if person_count == 0:
        print("Không có dữ liệu người quen nào. Vui lòng xử lý khuôn mặt đã biết trước!")
        input("Nhấn Enter để tiếp tục...")
        return
    else:
        print(f"Đã tải {person_count} người quen vào bộ nhớ")
    
    # Khởi động camera
    print("Đang khởi động camera...")
    cap = cv2.VideoCapture(0)
    
    # Thiết lập camera để cân bằng độ phân giải và FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Độ phân giải trung bình
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm độ trễ
    
    if not cap.isOpened():
        print("Không thể mở camera. Vui lòng kiểm tra lại kết nối.")
        input("Nhấn Enter để tiếp tục...")
        return
    
    print("Camera đã sẵn sàng. Nhấn 'q' để thoát, 's' để chụp ảnh màn hình")
    
    # Khởi tạo các biến theo dõi hiệu suất
    fps_start_time = time.time()
    frame_count = 0
    process_this_frame = 0
    fps = 0.0
    avg_processing_time = 0.0
    processing_times = []
    
    # Reset biến global
    global latest_recognition, processing
    latest_recognition = []
    processing = False
    recognition_buffers = {}
    
    # Thư mục lưu ảnh chụp màn hình
    screenshots_dir = os.path.join(os.getcwd(), "screenshots")
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)
    
    # Vòng lặp chính xử lý video
    while True:
        # Bắt đầu đo thời gian xử lý frame
        frame_start_time = time.time()
        
        # Đọc frame từ camera
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera. Thoát...")
            break

        # Tính FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 0.5:  # Cập nhật FPS 2 lần mỗi giây
            fps = frame_count / elapsed_time
            frame_count = 0
            fps_start_time = time.time()

        # Chỉ xử lý mỗi DETECTION_INTERVAL frame để cải thiện hiệu suất
        process_this_frame = (process_this_frame + 1) % DETECTION_INTERVAL
        
        # Tạo frame hiển thị với tỷ lệ khung hình đúng
        h, w = frame.shape[:2]
        display_height = int(h * DISPLAY_WIDTH / w)
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, display_height))
        
        # Xử lý phát hiện và nhận dạng khuôn mặt
        if process_this_frame == 0 and not processing:
            processing = True
            process_start = time.time()
            
            try:
                # Phát hiện khuôn mặt trên frame nhỏ hơn để tăng tốc
                detection_frame = cv2.resize(frame, (DETECTION_SIZE, DETECTION_SIZE))
                
                # Tăng tương phản để cải thiện khả năng phát hiện
                detection_frame = cv2.convertScaleAbs(detection_frame, alpha=1.3, beta=10)
                
                # Thực hiện phát hiện khuôn mặt
                with tf.device('/GPU:0') if gpus else tf.device('/CPU:0'):
                    faces = RetinaFace.detect_faces(detection_frame, threshold=0.9)
                
                # Xử lý các khuôn mặt được phát hiện
                if faces:
                    # Tỉ lệ để chuyển từ khung phát hiện sang khung gốc
                    scale_x = w / DETECTION_SIZE
                    scale_y = h / DETECTION_SIZE
                    
                    # Sắp xếp khuôn mặt theo diện tích giảm dần
                    sorted_faces = []
                    for face_key in faces:
                        face_data = faces[face_key]
                        facial_area = face_data['facial_area']
                        x1, y1, x2, y2 = facial_area
                        
                        # Scale về kích thước gốc
                        x1_orig = int(x1 * scale_x)
                        y1_orig = int(y1 * scale_y)
                        x2_orig = int(x2 * scale_x)
                        y2_orig = int(y2 * scale_y)
                        
                        # Đảm bảo tọa độ nằm trong frame
                        x1_orig = max(0, min(x1_orig, w-1))
                        y1_orig = max(0, min(y1_orig, h-1))
                        x2_orig = max(x1_orig+1, min(x2_orig, w))
                        y2_orig = max(y1_orig+1, min(y2_orig, h))
                        
                        area = (x2_orig - x1_orig) * (y2_orig - y1_orig)
                        sorted_faces.append((face_key, area, (x1_orig, y1_orig, x2_orig, y2_orig)))
                    
                    # Sắp xếp theo diện tích và giới hạn số khuôn mặt xử lý
                    sorted_faces.sort(key=lambda x: x[1], reverse=True)
                    face_results = []
                    
                    # Chỉ xử lý MAX_FACES khuôn mặt lớn nhất
                    for face_idx, (face_key, _, facial_area) in enumerate(sorted_faces[:MAX_FACES]):
                        x1, y1, x2, y2 = facial_area
                        
                        # Bỏ qua khuôn mặt quá nhỏ
                        if (x2 - x1) < 30 or (y2 - y1) < 30:
                            continue
                            
                        # Cắt khuôn mặt từ frame gốc
                        face_img = frame[y1:y2, x1:x2]
                        
                        # Tính hash nhanh cho việc cache
                        resized_face = cv2.resize(face_img, (16, 16))
                        face_hash = hash(resized_face.tobytes())
                        
                        # Nhận dạng khuôn mặt
                        name = "Unknown"
                        distance = None
                        
                        # Kiểm tra cache trước
                        cache_result = face_cache.get(face_hash)
                        if cache_result:
                            name, distance = cache_result
                        else:
                            # Trích xuất embedding chỉ khi cần
                            embedding = extract_embedding(face_img)
                            if embedding is not None:
                                name, distance = compare_faces(embedding, known_faces)
                                # Chỉ cache kết quả người quen để tiết kiệm bộ nhớ
                                if name != "Unknown":
                                    face_cache.put(face_hash, (name, distance))
                        
                        # Sử dụng buffer riêng cho mỗi vị trí khuôn mặt
                        buffer_key = f"face_{face_idx}"
                        if buffer_key not in recognition_buffers:
                            recognition_buffers[buffer_key] = RecognitionBuffer(size=RECOGNITION_BUFFER_SIZE)
                            
                        recognition_buffers[buffer_key].update(name, distance)
                        stable_name, stable_distance = recognition_buffers[buffer_key].get_result()
                        
                        # Chuyển đổi tọa độ sang frame hiển thị
                        display_x1 = int(x1 * DISPLAY_WIDTH / w)
                        display_y1 = int(y1 * display_height / h)
                        display_x2 = int(x2 * DISPLAY_WIDTH / w)
                        display_y2 = int(y2 * display_height / h)
                        
                        face_results.append(((display_x1, display_y1, display_x2, display_y2), 
                                           stable_name, stable_distance))
                    
                    # Cập nhật kết quả nhận diện
                    latest_recognition = face_results
                
                # Ghi lại thời gian xử lý để phân tích hiệu suất
                process_time = time.time() - process_start
                processing_times.append(process_time)
                if len(processing_times) > 30:  # Giữ thống kê 30 frame gần nhất
                    processing_times.pop(0)
                avg_processing_time = sum(processing_times) / len(processing_times)
            
            except Exception as e:
                print(f"Lỗi phát hiện khuôn mặt: {str(e)[:50]}...")
            
            processing = False
        
        # Hiển thị kết quả nhận dạng
        for facial_area, name, distance in latest_recognition:
            try:
                x1, y1, x2, y2 = facial_area
                
                # Xác định màu và nhãn dựa trên kết quả nhận dạng
                if name == "Unknown":
                    color = (0, 0, 255)  # Đỏ cho người lạ
                    label = "Unknown"
                else:
                    # Tính độ tin cậy với sự tăng cường

                    confidence_pct = min(100, max(70, int((1 - distance/THRESHOLD) * CONFIDENCE_BOOST * 100)))
                    
                    # Màu dựa trên độ tin cậy (từ xanh lá đến vàng)
                    if confidence_pct > 85:  # Tin cậy cao
                        color = (0, 255, 0)  # Xanh lá
                    elif confidence_pct > 75:  # Tin cậy trung bình
                        color = (0, 255, 255)  # Vàng
                    else:  # Tin cậy thấp
                        color = (0, 165, 255)  # Cam
                    
                    label = f"{name} ({confidence_pct}%)"
                
                # Vẽ khung xung quanh khuôn mặt
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Vẽ nền cho nhãn
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
                cv2.rectangle(display_frame, 
                              (x1, y1 - 30), 
                              (x1 + label_size[0] + 10, y1), 
                              color, cv2.FILLED)
                
                # Hiển thị nhãn
                cv2.putText(display_frame, label, (x1 + 5, y1 - 8), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception:
                pass
        
        # Hiển thị thống kê về hiệu năng
        try:
            # Tạo thanh trạng thái ở trên cùng
            status_height = 50
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], status_height), (0, 0, 0), cv2.FILLED)
            
            # FPS
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(display_frame, fps_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Thời gian xử lý
            time_text = f"Process: {avg_processing_time*1000:.1f}ms"
            cv2.putText(display_frame, time_text, (180, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Trạng thái GPU và model
            if gpus:
                gpu_text = "GPU: Active"
                color = (0, 255, 0)
            else:
                gpu_text = "GPU: Inactive"
                color = (0, 100, 255)
                
            cv2.putText(display_frame, gpu_text, (350, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            model_text = "Model: " + ("Facenet" if USE_LIGHTWEIGHT_MODEL else "ArcFace")
            cv2.putText(display_frame, model_text, (500, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Hiển thị số người đã biết
            cv2.putText(display_frame, f"DB: {person_count} người", 
                       (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        except Exception:
            pass
        
        # Hiển thị frame kết quả
        try:
            cv2.imshow("Face Recognition", display_frame)
        except Exception:
            break
        
        # Xử lý phím nhấn
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Thoát
            break
        elif key == ord('s'):  # Chụp màn hình
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            screenshot_path = os.path.join(screenshots_dir, f"capture_{timestamp}.jpg")
            cv2.imwrite(screenshot_path, display_frame)
            print(f"Đã lưu ảnh chụp màn hình tại: {screenshot_path}")

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

# Dọn dẹp thư mục tạm khi thoát
def cleanup_temp_files():
    if os.path.exists(TEMP_DIR):
        for file in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Lỗi khi xóa file tạm {file_path}: {e}")
    
    # Lưu disk cache
    try:
        face_cache.save_disk_cache()
    except:
        pass
    
    # Đóng các thread pool và process pool
    global PROCESS_POOL
    if PROCESS_POOL:
        PROCESS_POOL.shutdown(wait=False)

# Menu chính
def main():
    # Biến global cho debug mode
    global THRESHOLD
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Xóa màn hình
        print("===== HỆ THỐNG NHẬN DẠNG KHUÔN MẶT THỜI GIAN THỰC =====")
        print("1. Xử lý khuôn mặt đã biết")
        print("2. Bắt đầu nhận dạng thời gian thực")
        print("3. Thêm người quen mới bằng camera")
        print("4. Điều chỉnh ngưỡng nhận dạng")
        print("5. Kiểm tra dữ liệu người dùng")
        print("6. Kiểm tra môi trường và phụ thuộc")
        print("7. Thoát")
        print("=====================================================")
        
        choice = input("Chọn chức năng (1-7): ").strip()

        if choice == "1":
            process_known_faces()
        elif choice == "2":
            recognize_faces()
        elif choice == "3":
            add_new_person()
        elif choice == "4":
            try:
                current = THRESHOLD
                new_value = float(input(f"Ngưỡng hiện tại: {current}. Nhập ngưỡng mới (0.3-1.0): "))
                if 0.3 <= new_value <= 1.0:
                    THRESHOLD = new_value
                    print(f"Đã thiết lập ngưỡng mới: {THRESHOLD}")
                else:
                    print("Giá trị không hợp lệ. Ngưỡng phải nằm trong khoảng 0.3-1.0")
            except ValueError:
                print("Giá trị không hợp lệ. Vui lòng nhập một số thực.")
            input("Nhấn Enter để tiếp tục...")
        elif choice == "5":
            check_user_data()
        elif choice == "6":
            check_environment()
        elif choice == "7":
            print("Đang dọn dẹp tài nguyên...")
            cleanup_temp_files()
            print("Thoát chương trình. Cảm ơn bạn đã sử dụng!")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")
            input("Nhấn Enter để tiếp tục...")

# Thêm hàm kiểm tra môi trường và phụ thuộc
def check_environment():
    print("\n===== Kiểm tra môi trường và phụ thuộc =====")
    
    # Kiểm tra Python version
    import sys
    print(f"Python version: {sys.version}")
    
    # Kiểm tra các thư viện phụ thuộc
    libraries = {
        "TensorFlow": tf,
        "NumPy": np,
        "OpenCV": cv2,
        "RetinaFace": RetinaFace,
        "DeepFace": DeepFace
    }
    
    print("\nCác thư viện đã cài đặt:")
    for name, lib in libraries.items():
        try:
            version = lib.__version__
        except AttributeError:
            try:
                version = lib.version
            except AttributeError:
                version = "Không xác định"
        
        print(f"  - {name}: {version}")
    
    # Kiểm tra GPU
    print("\nThông tin GPU:")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i+1}: {gpu.name}")
        
        # Kiểm tra TensorFlow GPU
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                result = c.numpy()
            print("  - TensorFlow GPU hoạt động tốt")
        except Exception as e:
            print(f"  - Lỗi khi sử dụng TensorFlow GPU: {e}")
    else:
        print("  - Không tìm thấy GPU")
    
    # Kiểm tra đường dẫn
    print("\nCác đường dẫn:")
    paths = {
        "Known Faces": KNOWN_FACES_DIR,
        "Embeddings": EMBEDDINGS_DIR,
        "Temp": TEMP_DIR
    }
    
    for name, path in paths.items():
        exists = os.path.exists(path)
        status = "Tồn tại" if exists else "Không tồn tại"
        print(f"  - {name}: {path} ({status})")
    
    # Kiểm tra hệ điều hành
    import platform
    print(f"\nHệ điều hành: {platform.system()} {platform.release()}")
    
    # Hiển thị thông tin bộ nhớ
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"RAM: {vm.total / (1024**3):.2f} GB, Sử dụng: {vm.percent}%")
    except ImportError:
        print("Không thể hiển thị thông tin bộ nhớ (cần cài đặt psutil)")
    
    input("\nNhấn Enter để tiếp tục...")

# Hàm mới để kiểm tra dữ liệu người dùng
def check_user_data():
    print("\n===== Kiểm tra dữ liệu người dùng =====")
    
    # Kiểm tra thư mục ảnh
    print("\nThư mục ảnh người dùng:")
    users_with_images = 0
    total_images = 0
    
    if os.path.exists(KNOWN_FACES_DIR):
        for person_name in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
            if os.path.isdir(person_dir):
                images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                img_count = len(images)
                if img_count > 0:
                    users_with_images += 1
                    total_images += img_count
                    print(f"  - {person_name}: {img_count} ảnh")
                else:
                    print(f"  - {person_name}: Không có ảnh")
        
        print(f"\nTổng cộng: {users_with_images} người dùng có ảnh, {total_images} ảnh")
    else:
        print("  Thư mục ảnh không tồn tại!")
    
    # Kiểm tra embeddings
    print("\nEmbeddings đã xử lý:")
    users_with_embeddings = 0
    
    if os.path.exists(EMBEDDINGS_DIR):
        for file_name in os.listdir(EMBEDDINGS_DIR):
            if file_name.endswith(".npy"):
                users_with_embeddings += 1
                name = os.path.splitext(file_name)[0]
                file_path = os.path.join(EMBEDDINGS_DIR, file_name)
                file_size = os.path.getsize(file_path)
                
                try:
                    # Thử tải embedding để kiểm tra tính hợp lệ
                    embedding = np.load(file_path)
                    valid = "hợp lệ" if embedding.shape[0] > 0 else "không hợp lệ"
                    shape = embedding.shape
                except Exception as e:
                    valid = f"lỗi: {str(e)}"
                    shape = "không xác định"
                
                print(f"  - {name}: {file_size} bytes, {valid}, shape: {shape}")
        
        print(f"\nTổng cộng: {users_with_embeddings} người dùng có embedding")
    else:
        print("  Thư mục embeddings không tồn tại!")
    
    input("\nNhấn Enter để tiếp tục...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nChương trình đã bị dừng bởi người dùng.")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi: {e}")
        traceback.print_exc()
    finally:
        cleanup_temp_files()
        input("Nhấn Enter để thoát...")
