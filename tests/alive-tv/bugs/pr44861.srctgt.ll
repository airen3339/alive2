; Found by Alive2
declare void @g(ptr)

define void @src(ptr %ptr, i1 %cc1, i1 %cc2, i1 %cc3, i32 %arg) {
entry:
  br i1 %cc1, label %do.end, label %if.then

if.then:
  br i1 %cc2, label %do.body, label %do.end

do.body:
  %phi = phi i32 [ %arg, %if.then ]
  %phi.ext = zext i32 %phi to i64
  %ptr2 = getelementptr inbounds i8, ptr %ptr, i64 %phi.ext
  %ptr3 = getelementptr inbounds i8, ptr %ptr2, i64 -1
  call void @g(ptr %ptr3)
  br i1 %cc3, label %do.end, label %if.then

do.end:
  ret void
}

define void @tgt(ptr %ptr, i1 %cc1, i1 %cc2, i1 %cc3, i32 %arg) {
  br i1 %cc1, label %do.end, label %if.then

if.then:                                          ; preds = %do.body, %entry
  br i1 %cc2, label %do.body, label %do.end

do.body:                                          ; preds = %if.then
  %phi.ext = zext i32 %arg to i64
  %ptr2 = getelementptr inbounds i8, ptr %ptr, i64 -1
  %ptr3 = getelementptr inbounds i8, ptr %ptr2, i64 %phi.ext
  call void @g(ptr nonnull %ptr3)
  br i1 %cc3, label %do.end, label %if.then

do.end:                                           ; preds = %do.body, %if.then, %entry
  ret void
}

; ERROR: Source is more defined than target
