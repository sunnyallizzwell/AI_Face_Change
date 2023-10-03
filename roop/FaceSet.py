import numpy as np

class FaceSet:
    faces = []
    embedding_average = 'None'
    embeddings_backup = None

    def __init__(self):
        self.faces = []
        self.embedding_average = 'None'
        self.embeddings_backup = None

    def AverageEmbeddings(self):
        if len(self.faces) > 1 and self.embeddings_backup is None:
            self.embeddings_backup = self.faces[0]['embedding']
            embeddings_average = []

            for face in self.faces:
                embeddings_average.append(face['embedding'])

            self.faces[0]['embedding'] = np.mean(embeddings_average, axis=0)
            # self.faces[0]['embedding'] = np.median(embeddings_average, axis=0)
