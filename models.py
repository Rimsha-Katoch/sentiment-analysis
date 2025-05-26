from django.db import models

class Comment(models.Model):
    comment = models.TextField()
    prediction = models.CharField(max_length=20)  # "Suspect" or "Non-Suspect"
    confidence = models.FloatField(default=0.0)  # Confidence score as percentage
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.prediction} ({self.confidence:.1f}%): {self.comment[:50]}"
