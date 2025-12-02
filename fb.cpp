#include <GL/glew.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <memory.h>

#include "fb.h"

struct Framebuffer {
    GLuint fbo;
    GLuint tex_id;
    int w,h;
    GLenum fmt;
};

// Function to check framebuffer status
static bool checkFramebufferStatus(GLuint fbo) {
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        printf("Framebuffer is not complete: ");
        switch (status) {
            case GL_FRAMEBUFFER_UNDEFINED:
                printf("GL_FRAMEBUFFER_UNDEFINED\n");
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                printf("GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT\n");
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                printf("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT\n");
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
                printf("GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER\n");
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
                printf("GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER\n");
                break;
            case GL_FRAMEBUFFER_UNSUPPORTED:
                printf("GL_FRAMEBUFFER_UNSUPPORTED\n");
                break;
            default:
                printf("Unknown error 0x%x\n", status);
        }
        return false;
    } else {
        return true;
    }
}


Framebuffer* FB_Create(int w, int h) {
    Framebuffer* pfb = (Framebuffer*)malloc(sizeof(Framebuffer));
    Framebuffer& fb = *pfb;
    fb.w = w;
    fb.h = h;

    glGenFramebuffers(1, &fb.fbo);

    glGenTextures(1, &fb.tex_id);
    glBindTexture(GL_TEXTURE_2D, fb.tex_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // attach texture to fbo
    glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);  
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fb.tex_id, 0);

    if(!checkFramebufferStatus(fb.fbo)) {
        fprintf(stderr, "Failed to create Framebuffer\n");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);  

    return pfb;
}

void FB_Draw(const Framebuffer& fb, FB_DrawCB_t cb, void* puserdata) {
    
    glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // we're not using the stencil buffer now
    glEnable(GL_DEPTH_TEST);

    cb(puserdata);	

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FB_Destroy(Framebuffer* fb) {

    glBindFramebuffer(GL_FRAMEBUFFER, 0);  
    glDeleteFramebuffers(1, &fb->fbo);

    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, &fb->tex_id);  

    free(fb);
}

//--------------------------------------------------------------------------------------------------------------------------------

struct DynamicTexture {
    GLuint tex_id;
    GLuint pbo_id;
    GLenum fmt;
    int w, h;
};

DynamicTexture* DT_Create(int w, int h) {
    DynamicTexture* pdt = (DynamicTexture*)malloc(sizeof(DynamicTexture));
    DynamicTexture& dt = *pdt;
    dt.w = w;
    dt.h = h;
    dt.fmt = GL_BGRA;

    glGenTextures(1, &dt.tex_id);
    glBindTexture(GL_TEXTURE_2D, dt.tex_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, dt.fmt, GL_UNSIGNED_BYTE, (GLvoid*)0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenBuffers(1, &dt.pbo_id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, dt.pbo_id);
    GLsizeiptr pboSize = w * h * 4 * sizeof(unsigned char);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, pboSize, NULL, GL_STREAM_DRAW);

    return pdt;
}

// TODO: use 2 PBO: https://www.songho.ca/opengl/gl_pbo.html 
void DT_Update(const DynamicTexture* dt, void* data, int size) {
    assert(size == dt->w*dt->h*4);

    glBindTexture(GL_TEXTURE_2D, dt->tex_id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, dt->pbo_id);

    // copy pixels from PBO to texture object. Use offset instead of ponter.
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, dt->w, dt->h, dt->fmt, GL_UNSIGNED_BYTE, 0);

    // bind PBO to update pixel values
    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboIds[nextIndex]);

    glBufferData(GL_PIXEL_UNPACK_BUFFER, size, 0, GL_STREAM_DRAW);
    GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    if(ptr)
    {
        memcpy(ptr, data, size);
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);  // release pointer to mapping buffer
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void DT_Update(const struct DynamicTexture* dt, void* puserdata, DT_UpdateCB_t cb) {

    glBindTexture(GL_TEXTURE_2D, dt->tex_id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, dt->pbo_id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, dt->w, dt->h, dt->fmt, GL_UNSIGNED_BYTE, 0);

    glBufferData(GL_PIXEL_UNPACK_BUFFER, dt->w*dt->h*4, 0, GL_STREAM_DRAW);
    GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    cb(ptr, puserdata);
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
                                            
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

unsigned int DT_GetTextureID(DynamicTexture* dt) { 
    return dt->tex_id;
}

unsigned int DT_GetWidth(DynamicTexture* dt) { 
    return dt->w;
}

unsigned int DT_GetHeight(DynamicTexture* dt) { 
    return dt->h;
}

void DT_Destroy(DynamicTexture* dt) {

    glDeleteTextures(1, &dt->tex_id);
    glDeleteBuffers(1, &dt->pbo_id);
}

