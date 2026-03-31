from django.db import models

class ChatQuery(models.Model):
    session_key = models.CharField(max_length=40, db_index=True)
    prompt = models.TextField()
    response = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.prompt[:50]}..."
    
    class Meta:
        ordering = ['-created_at']
